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
                self.embedder.embed_categories(
                    conn,
                    DataStore.CATEGORY_TAB,
                    DataStore.CATEGORY_EMBEDS)
                self.embedder.embed_reviews(conn, DataStore.REVIEW_TAB)
                self.embedder.embed_podcasts(
                    conn,
                    DataStore.PODCAST_TAB,
                    DataStore.PODCAST_EMBEDS)
            else:
                if not DataStore._assert_categories(conn):
                    DataStore._prep_categories(conn)
                    self.embedder.embed_categories(
                        conn,
                        DataStore.CATEGORY_TAB,
                        DataStore.CATEGORY_EMBEDS)
                if not DataStore._assert_reviews(conn):
                    DataStore._prep_reviews(conn)
                    self.embedder.embed_reviews(conn, DataStore.REVIEW_TAB)
                if not DataStore._assert_podcasts(conn):
                    DataStore._prep_podcasts(conn)
                    self.embedder.embed_podcasts(
                        conn,
                        DataStore.PODCAST_TAB,
                        DataStore.PODCAST_EMBEDS)

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
            raise e
