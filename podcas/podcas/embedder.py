from logging import getLogger
from duckdb import DuckDBPyConnection
from sentence_transformers import SentenceTransformer


# TODO: decouple Embedder from DB
# and move db logic to DataStore
class Embedder:
    _logger = getLogger(f"{__name__}.{__qualname__}")

    def __init__(
            self,
            category_model: str,
            review_model: str,
            podcast_model: str
    ):
        self.category_embedder = SentenceTransformer(category_model)
        self.review_embedder = SentenceTransformer(review_model)
        self.podcast_embedder = SentenceTransformer(podcast_model)
        self.models = {
            'category': category_model,
            'review': review_model,
            'podcast': podcast_model
        }

    def embed_categories(
            self,
            conn: DuckDBPyConnection,
            table: str,
            embedded_table: str
    ) -> None:
        query = f'SELECT DISTINCT category FROM {table}'
        categories: list[str] = [
            result[0]
            for result in conn.sql(query).fetchall()
        ]

        Embedder._logger.info("Embedding categories...")
        embeddings = self.category_embedder.encode(
            categories,
            show_progress_bar=True
        )
        category_index = {
            category: embedding.tolist()
            for category, embedding in zip(categories, embeddings)
        }
        dim = len(category_index[categories[0]])

        conn.begin()
        try:
            conn.sql(f"DROP TABLE IF EXISTS {embedded_table}")
            conn.sql(f"""
            CREATE TABLE {embedded_table}(
                name VARCHAR PRIMARY KEY,
                vec FLOAT[{dim}]
            )
            """)

            # # TODO: partition or parallelize over cursors
            conn.executemany(
                f"INSERT INTO {embedded_table} VALUES (?, ?)",
                [
                    [category, vector]
                    for category, vector in category_index.items()
                ]
            )

            conn.sql(f"""
            CREATE INDEX idx_cat
            ON {embedded_table} USING HNSW (vec)
            WITH (metric = 'cosine')
            """)

            conn.commit()

        except Exception as e:
            Embedder._logger.error(f'Error occured during transaction: {e}', exc_info=True)
            Embedder._logger.debug('Rolling back...')
            conn.rollback()
            raise e

    def embed_reviews(self, conn: DuckDBPyConnection, table: str) -> None:
        reviews: list[tuple[str, str, str]] = conn.sql(f"""
        SELECT rev_id, title, content
        FROM {table}
        """).fetchall()
        ids = [idx for idx, _, _ in reviews]
        aggregated = [
            f'TITLE:{title} - CONTENT:{content}'
            for _, title, content in reviews
        ]

        Embedder._logger.info("Embedding review titles...")
        title_embeddings = self.review_embedder.encode(
            [title for _, title, _ in reviews],
            show_progress_bar = True
        )

        Embedder._logger.info("Embedding review content...")
        content_embeddings = self.review_embedder.encode(
            [content for _, _, content in reviews],
            show_progress_bar = True
        )

        Embedder._logger.info("Embedding full reviews...")
        review_embeddings = self.review_embedder.encode(
            aggregated,
            show_progress_bar = True
        )
        dim = title_embeddings[0].shape[0]

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

        conn.begin()
        try:
            conn.sql(f"ALTER TABLE {table} DROP COLUMN IF EXISTS vec_title")
            conn.sql(f"ALTER TABLE {table} ADD COLUMN vec_title FLOAT[{dim}]")

            conn.sql(f"ALTER TABLE {table} DROP COLUMN IF EXISTS vec_content")
            conn.sql(f"ALTER TABLE {table} ADD COLUMN vec_content FLOAT[{dim}]")

            conn.sql(f"ALTER TABLE {table} DROP COLUMN IF EXISTS vec_aggregated")
            conn.sql(f"ALTER TABLE {table} ADD COLUMN vec_aggregated FLOAT[{dim}]")

            # TODO: partition or parallelize over cursors
            conn.sql(f"""
            UPDATE {table}
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
            ON {table} USING HNSW (vec_title)
            WITH (metric = 'cosine')
            """)
            conn.sql(f"""
            CREATE INDEX idx_rev_content
            ON {table} USING HNSW (vec_content)
            WITH (metric = 'cosine')
            """)
            conn.sql(f"""
            CREATE INDEX idx_rev_aggregated
            ON {table} USING HNSW (vec_aggregated)
            WITH (metric = 'cosine')
            """)

            conn.commit()

        except Exception as e:
            Embedder._logger.error(f'Error occured during transaction: {e}', exc_info=True)
            Embedder._logger.debug('Rolling back...')
            conn.rollback()
            raise e

    def embed_podcasts(
            self,
            conn: DuckDBPyConnection,
            table: str,
            embedded_table: str
    ) -> None: ...
