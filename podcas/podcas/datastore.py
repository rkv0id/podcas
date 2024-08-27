from contextlib import contextmanager
from typing import Generator
from os.path import abspath
import logging, duckdb

from .embedder import Embedder


class DataStore:
    CATEGORY_TAB = 'categories'
    PODCAST_TAB = 'podcasts'
    REVIEW_TAB = 'reviews'
    CATEGORY_EMBEDS = 'v_cat'
    PODCAST_EMBEDS = 'v_pod'

    _logger = logging.getLogger(f"{__name__}.{__qualname__}")

    def __init__(self, path: str):
        self.file = abspath(path)
        with self._conn() as db_conn:
            if not DataStore._assert_categories(db_conn):
                Embedder._embed_categories(
                    db_conn,
                    DataStore.CATEGORY_TAB,
                    DataStore.CATEGORY_EMBEDS)
            if not DataStore._assert_reviews(db_conn):
                Embedder._embed_reviews(db_conn, DataStore.REVIEW_TAB)
            Embedder._embed_podcasts(db_conn, DataStore.PODCAST_TAB)

    @contextmanager
    def _conn(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
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
        AND column_name IN ('vec_title', 'vec_content', 'vec_aggregated')
        """
        result, *_ = conn.sql(query).fetchall()
        return result[0] == 3
