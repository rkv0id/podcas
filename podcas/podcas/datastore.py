from contextlib import contextmanager
from typing import Generator
from os.path import abspath
from sentence_transformers import SentenceTransformer
import duckdb

class _DataStore:
    def __init__(self, path: str):
        self._file = abspath(path)
        with self.conn() as db_conn:
            if not _DataStore.check_embeddings_exist(db_conn):
                # tqdm db prep
                self.preprocess()
                self.embed()
                self.store()

    @contextmanager
    def conn(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        connection = duckdb.connect(self._file)
        try: yield connection
        except Exception as e:
            print(f'Could not obtain database connection curosr: {e}')
        finally:
            connection.close()

    def preprocess(self): ...
    def embed(self): ...
    def store(self): ...

    @staticmethod
    def check_embeddings_exist(conn: duckdb.DuckDBPyConnection) -> bool:
        required_tables = {'categories', 'reviews', 'podcasts'}
        tablenames_str = "', '".join(required_tables)

        query = f"""
        SELECT COUNT(table_name)
        FROM information_schema.tables
        WHERE lower(table_name) IN ('{tablenames_str}')
        """

        result, *_ = conn.execute(query).fetchall()
        return result[0] == 3
