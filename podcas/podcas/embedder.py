from logging import getLogger
from duckdb import DuckDBPyConnection
from sentence_transformers import SentenceTransformer


class Embedder:
    _logger = getLogger(f"{__name__}.{__qualname__}")

    @staticmethod
    def _embed_categories(
            conn: DuckDBPyConnection,
            table: str,
            embedded_table: str
    ) -> None:
        query = f'SELECT DISTINCT category FROM {table}'
        categories: list[str] = [result[0] for result in conn.sql(query).fetchall()]
        encoder = SentenceTransformer('all-mpnet-base-v2')

        Embedder._logger.info("Embedding categories...")
        embeddings = encoder.encode(categories, show_progress_bar=True)
        category_index = {
            category: embedding
            for category, embedding in zip(categories, embeddings)
        }
        dim = category_index[categories[0]].shape[0]

        conn.begin()
        try:
            conn.sql(f"""
            CREATE TABLE {embedded_table}(
                name VARCHAR PRIMARY KEY,
                vec FLOAT[{dim}]
            )
            """)

            for category, vector in category_index.items():
                vector_str = ', '.join([str(x) for x in vector.tolist()])
                conn.sql(f"""
                INSERT INTO {embedded_table}
                VALUES ('{category}', [{vector_str}])
                """)

            conn.sql(f"""
            CREATE INDEX idx_cat
            ON {embedded_table} USING HNSW (vec)
            WITH (metric = 'cosine')
            """)

            conn.commit()

        except Exception as e:
            Embedder._logger.error(f'Error occured during transaction: {e}')
            Embedder._logger.error('Rolling back...')
            conn.rollback()

    @staticmethod
    def _embed_reviews(conn: DuckDBPyConnection, table: str) -> None:
        conn.begin()
        try:
            conn.sql(f"""
            ALTER TABLE {table}
            ADD COLUMN rev_id INTEGER
            """)

            conn.sql(f"""
            UPDATE {table}
            SET rev_id = seq.seq_num
            FROM (
                SELECT ROW_NUMBER() OVER () AS seq_num, rowid
                FROM {table}
            ) AS seq
            WHERE {table}.rowid = seq.rowid
            """)

            reviews: list[str] = conn.sql(f"""
            SELECT rev_id, title, content
            FROM {table}
            """).fetchall()
            ids = [idx for idx, _, _ in reviews]
            aggregated = [
                f'TITLE:{title} - CONTENT:{content}'
                for _, title, content in reviews
            ]

            encoder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
            Embedder._logger.info("Embedding review titles...")
            title_embeddings = encoder.encode(
                [title for _, title, _ in reviews],
                show_progress_bar = True
            )

            Embedder._logger.info("Embedding review content...")
            content_embeddings = encoder.encode(
                [content for _, _, content in reviews],
                show_progress_bar = True
            )

            Embedder._logger.info("Embedding full reviews...")
            review_embeddings = encoder.encode(aggregated, show_progress_bar = True)
            dim = title_embeddings[0].shape[0]

            conn.sql(f"""
            ALTER TABLE {table}
            ADD COLUMN vec_title FLOAT[{dim}]
            """)
            conn.sql(f"""
            ALTER TABLE {table}
            ADD COLUMN vec_content FLOAT[{dim}]
            """)
            conn.sql(f"""
            ALTER TABLE {table}
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
            # to be parallelised over partitions and/or cursors
            # a column-oriented db would ve been much better
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
            Embedder._logger.error(f'Error occured during transaction: {e}')
            Embedder._logger.error('Rolling back...')
            conn.rollback

    @staticmethod
    def _embed_podcasts(conn: DuckDBPyConnection, table: str) -> None: ...
