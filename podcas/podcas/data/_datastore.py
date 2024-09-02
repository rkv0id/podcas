from contextlib import contextmanager
from typing import Any, Generator, Optional
from os.path import abspath
import logging, duckdb

from podcas.ml import Embedder, Mooder


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
            embedder: Embedder,
            mooder: Mooder
    ):
        self.file = abspath(path)
        self.embedder = embedder
        self.mooder = mooder
        self.model_names = {
            **self.embedder.model_names,
            'sentiment': self.mooder.model_name,
            'sentiment_summary': (
                self.mooder.summarizer.model_name
                if self.mooder.summarizer else "NO-OP"
            ),
            'embedding_summary': (
                self.embedder.summarizer.model_name
                if self.embedder.summarizer else "NO-OP"
            )
        }
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            if not self._assert_meta(conn):
                self._record_meta(conn)
                self._prep_categories(conn)
                self._prep_reviews(conn)
                self._prep_episodes(conn)
                self._prep_podcasts(conn)
                self._embed_categories(conn)
                self._embed_reviews(conn)
                self._embed_episodes(conn)
                self._embed_podcasts(conn)
            else:
                if not DataStore._assert_categories(conn):
                    self._prep_categories(conn)
                    self._embed_categories(conn)
                if not DataStore._assert_reviews(conn):
                    self._prep_reviews(conn)
                    self._embed_reviews(conn)
                if not DataStore._assert_episodes(conn):
                    self._prep_episodes(conn)
                    self._embed_episodes(conn)
                if not DataStore._assert_podcasts(conn):
                    self._prep_podcasts(conn)
                    self._embed_podcasts(conn)

    def get_reviews(
            self,
            top: int,
            rating_range: tuple[float, float],
            sentiment: Optional[str],
            query_embedding: Optional[list[float]],
            rating_boost: bool
    ) -> list[tuple[str, str, float, float]]:
        query = "SELECT title, content, rating,"

        if not query_embedding: query += ' 1'
        else:
            score = f"""
            (
                1 + array_cosine_similarity(vec,
                {query_embedding}::FLOAT[{len(query_embedding)}])
            ) / 2
            """
            if rating_boost: score += "* rating / 5.0"
            query += score

        query += f"""
        AS score
        FROM {DataStore.REVIEW_TAB}
        WHERE rating BETWEEN {rating_range[0]} AND {rating_range[1]}
        {"AND embedded" if query_embedding else ""}
        {f"AND sentiment = '{sentiment}'" if sentiment else ""}
        ORDER BY score DESC, rating DESC
        LIMIT {top}
        """

        with self._conn() as conn:
            try: result = conn.sql(query).fetchall()
            except duckdb.ProgrammingError as error:
                DataStore._logger.error(
                    f"Input-caused error while getting reviews: {error}",
                    exc_info=True
                )
                result = []

        return result

    def get_episodes(
            self,
            top: int,
            rating_range: tuple[float, float],
            title: Optional[str],
            fuzzy_title: bool,
            author: Optional[str],
            fuzzy_author: bool,
            review_embeddings: Optional[list[float]],
            desc_embeddings: Optional[list[float]],
            rating_boost: bool
    ) -> list[tuple[str, str, str, float, float]]:
        query = """
        SELECT title, author, itunes_id,
            CASE WHEN average_rating IS NULL THEN 0
            ELSE average_rating END AS rating,
        """

        if not (review_embeddings or desc_embeddings): query += ' 1'
        else:
            scores = []

            if review_embeddings:
                scores.append(f"""
                    (
                        1 + array_cosine_similarity(vec_rev,
                        {review_embeddings}::FLOAT[{len(review_embeddings)}])
                    ) / 2
                """)

            if desc_embeddings:
                scores.append(f"""
                    (
                        1 + array_cosine_similarity(vec_desc,
                        {desc_embeddings}::FLOAT[{len(desc_embeddings)}])
                    ) / 2
                """)

            query += '(' + ' + '.join(scores) + ')'
            if rating_boost:
                query += f"""
                * (CASE WHEN rating = 0 THEN 2.5 ELSE rating END)
                / {len(scores) * 5.}
                """

        if title:
            title_search_str = '%'.join(title) if fuzzy_title else title
            title_search_str = "'%" + title_search_str.replace("'", "''") + "%'"

        if author:
            author_search_str = '%'.join(author) if fuzzy_author else author
            author_search_str = "'%" + author_search_str.replace("'", "''") + "%'"

        query += f"""
        AS score
        FROM {DataStore.PODCAST_TAB}
        WHERE rating BETWEEN {rating_range[0]} AND {rating_range[1]}
        {"AND title ILIKE " + title_search_str if title else ""}
        {"AND author ILIKE " + author_search_str if author else ""}
        {"AND embedded_rev" if review_embeddings else ""}
        {"AND embedded_desc" if desc_embeddings else ""}
        ORDER BY score DESC, rating DESC
        {", levenshtein(title, '" + title.replace("'", "''") + "') ASC" if title else ""}
        {", levenshtein(author, '" + author.replace("'", "''") + "') ASC" if author else ""}
        LIMIT {top}
        """

        with self._conn() as conn: result = conn.sql(query).fetchall()

        return result

    def get_podcasts(
            self,
            top: int,
            rating_range: tuple[float, float],
            title: Optional[str],
            fuzzy_title: bool,
            author: Optional[str],
            fuzzy_author: bool,
            category_embeddings: Optional[list[float]],
            desc_embeddings: Optional[list[float]],
            rating_boost: bool
    ) -> list[tuple[str, str, float, float]]:
        query = "SELECT title, author, rating,"

        if not (category_embeddings or desc_embeddings): query += ' 1'
        else:
            scores = []

            if category_embeddings:
                scores.append(f"""
                    (
                        1 + array_cosine_similarity(vec_cat,
                        {category_embeddings}::FLOAT[{len(category_embeddings)}])
                    ) / 2
                """)

            if desc_embeddings:
                scores.append(f"""
                    (
                        1 + array_cosine_similarity(vec_desc,
                        {desc_embeddings}::FLOAT[{len(desc_embeddings)}])
                    ) / 2
                """)

            query += '(' + ' + '.join(scores) + ')'
            if rating_boost:
                query += f"""
                * (CASE WHEN rating = 0 THEN 2.5 ELSE rating END)
                / {len(scores) * 5.}
                """

        if title:
            title_search_str = '%'.join(title) if fuzzy_title else title
            title_search_str = "'%" + title_search_str.replace("'", "''") + "%'"

        if author:
            author_search_str = '%'.join(author) if fuzzy_author else author
            author_search_str = "'%" + author_search_str.replace("'", "''") + "%'"

        query += f"""
        AS score
        FROM {DataStore.PODCAST_EMBEDS}
        WHERE rating BETWEEN {rating_range[0]} AND {rating_range[1]}
        {"AND title ILIKE " + title_search_str if title else ""}
        {"AND author ILIKE " + author_search_str if author else ""}
        {"AND embedded_cat" if category_embeddings else ""}
        {"AND embedded_desc" if desc_embeddings else ""}
        ORDER BY score DESC, rating DESC
        {", levenshtein(title, '" + title.replace("'", "''") + "') ASC" if title else ""}
        {", levenshtein(author, '" + author.replace("'", "''") + "') ASC" if author else ""}
        LIMIT {top}
        """

        with self._conn() as conn: result = conn.sql(query).fetchall()

        return result

    def _embed_categories(self, conn: duckdb.DuckDBPyConnection) -> None:
        query = f"""
        SELECT DISTINCT category
        FROM {DataStore.CATEGORY_TAB}
        WHERE category IS NOT NULL"""

        categories: list[str] = [
            result[0]
            for result in conn.sql(query).fetchall()
        ]

        category_embeddings = self.embedder.embed_categories(categories)

        DataStore._logger.info('Ingesting categories embeddings...')
        if categories:
            conn.executemany(
                f"""
                INSERT INTO {DataStore.CATEGORY_EMBEDS} VALUES (?, ?)""",
                [
                    [category, vector]
                    for category, vector in category_embeddings.items()
                ]
            )

        DataStore._logger.info('Indexing categories vector space...')
        conn.sql(f"""
        CREATE INDEX idx_cat
        ON {DataStore.CATEGORY_EMBEDS} USING HNSW (vec)
        WITH (metric = 'cosine')""")

    def _embed_reviews(self, conn: duckdb.DuckDBPyConnection) -> None:
        reviews: list[tuple[int, str, str]] = conn.sql(f"""
        SELECT rev_id, title, content
        FROM {DataStore.REVIEW_TAB}
        WHERE title IS NOT NULL OR content IS NOT NULL
        """).fetchall()

        ids = [idx for idx, _, _ in reviews]
        review_embeddings = self.embedder.embed_reviews(reviews)
        review_sentiments = self.mooder.analyze_reviews(reviews)

        DataStore._logger.info('Ingesting reviews embeddings...')
        if ids:
            conn.executemany(f"""
            UPDATE {DataStore.REVIEW_TAB}
            SET vec = ?, embedded = ?, sentiment = ?
            WHERE rev_id = ?""", [
                [embed, 'true', sentiment.lower(), idx]
                for embed, sentiment, idx
                in zip(review_embeddings, review_sentiments, ids)
            ])

        DataStore._logger.info('Indexing reviews vector space...')
        conn.sql(f"""
        CREATE INDEX idx_rev
        ON {DataStore.REVIEW_TAB} USING HNSW (vec)
        WITH (metric = 'cosine')""")

    def _embed_episodes(self, conn: duckdb.DuckDBPyConnection) -> None:
        descriptions: list[tuple[str, str]] = conn.sql(f"""
        SELECT podcast_id, description
        FROM {DataStore.PODCAST_TAB}
        WHERE description IS NOT NULL
        """).fetchall()

        nested_rev_vectors = conn.execute(f"""
        SELECT p.podcast_id, LIST(r.vec)
        FROM {DataStore.PODCAST_TAB} p
        JOIN {DataStore.REVIEW_TAB} r
        ON p.podcast_id = r.podcast_id
        WHERE r.embedded
        GROUP BY p.podcast_id""").fetchall()

        desc_ids = [idx for idx, _ in descriptions]
        rev_ids = [idx for idx, _ in nested_rev_vectors]
        desc_embeddings, rev_embeddings = self.embedder.embed_episodes(
            [desc for _, desc in descriptions],
            [reviews for _, reviews in nested_rev_vectors]
        )

        if desc_ids:
            DataStore._logger.info('Ingesting episodes description-based embeddings...')
            conn.executemany(f"""
            UPDATE {DataStore.PODCAST_TAB}
            SET vec_desc = ?, embedded_desc = ?
            WHERE podcast_id = ?""", [
                [embed, 'true', idx]
                for embed, idx
                in zip(desc_embeddings, desc_ids)
            ])

        if rev_ids:
            DataStore._logger.info('Ingesting episodes review-based embeddings...')
            conn.executemany(f"""
            UPDATE {DataStore.PODCAST_TAB}
            SET vec_rev = ?, embedded_rev = ?
            WHERE podcast_id = ?""", [
                [embed, 'true', idx]
                for embed, idx
                in zip(rev_embeddings, rev_ids)
            ])

        DataStore._logger.info('Indexing episodes vector space...')
        DataStore._with_transaction(conn, [
            f"""
            CREATE INDEX idx_ep_desc
            ON {DataStore.PODCAST_TAB} USING HNSW (vec_desc)
            WITH (metric = 'cosine')""",
            f"""
            CREATE INDEX idx_ep_rev
            ON {DataStore.PODCAST_TAB} USING HNSW (vec_rev)
            WITH (metric = 'cosine')""",
        ])

    def _embed_podcasts(self, conn: duckdb.DuckDBPyConnection) -> None:
        nested_desc_vectors = conn.execute(f"""
        SELECT title, author, LIST(vec_desc)
        FROM {DataStore.PODCAST_TAB}
        WHERE embedded_desc
        AND title IS NOT NULL AND author IS NOT NULL
        GROUP BY title, author
        """).fetchall()

        nested_cat_vectors = conn.execute(f"""
        SELECT p.title, p.author, LIST(c.category)
        FROM {DataStore.PODCAST_TAB} p
        JOIN {DataStore.CATEGORY_TAB} c
        ON p.podcast_id = c.podcast_id
        WHERE p.title IS NOT NULL AND p.author IS NOT NULL
        GROUP BY p.author, p.title""").fetchall()

        desc_embeds, cat_embeds = self.embedder.embed_podcasts(
            [descriptions for _, _, descriptions in nested_desc_vectors],
            [categories for _, _, categories in nested_cat_vectors]
        )

        if nested_desc_vectors:
            title_author_desc_couples = [
                (
                    f"""{title.replace("'", "''")}""",
                    f"""{author.replace("'", "''")}""")
                for title, author, _ in nested_desc_vectors
            ]

            DataStore._logger.info("Ingesting podcasts description-based embeddings...")
            conn.executemany(f"""
            UPDATE {DataStore.PODCAST_EMBEDS}
            SET vec_desc = ?, embedded_desc = ?
            WHERE title = ? AND author = ?""", [
                [embed, 'true', couple[0], couple[1]]
                for embed, couple
                in zip(desc_embeds, title_author_desc_couples)
            ])

        if nested_cat_vectors:
            title_author_cat_couples = [
                (
                    f"""{title.replace("'", "''")}""",
                    f"""{author.replace("'", "''")}""")
                for title, author, _ in nested_cat_vectors
            ]

            DataStore._logger.info("Ingesting podcasts category-based embeddings...")
            conn.executemany(f"""
            UPDATE {DataStore.PODCAST_EMBEDS}
            SET vec_cat = ?, embedded_cat = ?
            WHERE title = ? AND author = ?""", [
                [embed, 'true', couple[0], couple[1]]
                for embed, couple
                in zip(cat_embeds, title_author_cat_couples)
            ])

        DataStore._with_transaction(conn, [
            f"""
            CREATE INDEX idx_pod_desc
            ON {DataStore.PODCAST_EMBEDS} USING HNSW (vec_desc)
            WITH (metric = 'cosine')""",
            f"""
            CREATE INDEX idx_pod_cat
            ON {DataStore.PODCAST_EMBEDS} USING HNSW (vec_cat)
            WITH (metric = 'cosine')"""
        ])

    def _assert_meta(self, conn: duckdb.DuckDBPyConnection) -> bool:
        table_exists = conn.execute(f"""
            SELECT COUNT(table_name)
            FROM information_schema.tables
            WHERE table_name = ?
        """, [DataStore.META_TAB]).fetchone()

        if not table_exists or table_exists[0] < 1: return False

        models_exist = None
        try:
            models_exist = conn.execute(f"""
            SELECT COUNT(*) FROM {DataStore.META_TAB}
            WHERE mod_cat = $category
            AND mod_rev = $review
            AND mod_pod = $podcast
            AND mod_sentiment = $sentiment
            AND emb_summary = $embedding_summary
            AND sent_summary = $sentiment_summary
            """, self.model_names).fetchone()
        except: return False

        return models_exist is not None and models_exist[0] == 1

    def _record_meta(self, conn: duckdb.DuckDBPyConnection) -> None:
        DataStore._with_transaction(conn, [
            f"DROP TABLE IF EXISTS {DataStore.META_TAB}",
            f"""
            CREATE TABLE {DataStore.META_TAB}(
                mod_cat VARCHAR,
                mod_rev VARCHAR,
                mod_pod VARCHAR,
                mod_sentiment VARCHAR,
                emb_summary VARCHAR,
                sent_summary VARCHAR)""",
            (
                f"""
                INSERT INTO {DataStore.META_TAB} VALUES (
                $category, $review, $podcast, $sentiment,
                $embedding_summary, $sentiment_summary)""",
                self.model_names
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
        existing = result[0] == 1

        query = f"""
        SELECT COUNT(column_name)
        FROM information_schema.columns
        WHERE table_name = '{DataStore.CATEGORY_EMBEDS}'
        AND column_name IN ('name', 'vec')
        """
        result, *_ = conn.sql(query).fetchall()
        complete = result[0] == 2

        return existing and complete

    @staticmethod
    def _assert_reviews(conn: duckdb.DuckDBPyConnection) -> bool:
        query = f"""
        SELECT COUNT(table_name)
        FROM information_schema.tables
        WHERE table_name = '{DataStore.REVIEW_TAB}'
        """
        result, *_ = conn.sql(query).fetchall()
        existing = result[0] == 1

        query = f"""
        SELECT COUNT(column_name)
        FROM information_schema.columns
        WHERE table_name = '{DataStore.REVIEW_TAB}'
        AND column_name IN ('rev_id', 'vec', 'embedded', 'sentiment')
        """
        result, *_ = conn.sql(query).fetchall()
        complete = result[0] == 4

        return existing and complete

    @staticmethod
    def _assert_episodes(conn: duckdb.DuckDBPyConnection) -> bool:
        query = f"""
        SELECT COUNT(table_name)
        FROM information_schema.tables
        WHERE table_name = '{DataStore.PODCAST_TAB}'
        """
        result, *_ = conn.sql(query).fetchall()
        existing = result[0] == 1

        query = f"""
        SELECT COUNT(column_name)
        FROM information_schema.columns
        WHERE table_name = '{DataStore.PODCAST_TAB}'
        AND column_name IN ('vec_desc', 'embedded_desc',
        'ratings_count_value', 'vec_rev', 'embedded_rev')
        """
        result, *_ = conn.sql(query).fetchall()
        complete = result[0] == 5

        return existing and complete

    @staticmethod
    def _assert_podcasts(conn: duckdb.DuckDBPyConnection) -> bool:
        query = f"""
        SELECT COUNT(table_name)
        FROM information_schema.tables
        WHERE table_name = '{DataStore.PODCAST_EMBEDS}'
        """
        result, *_ = conn.sql(query).fetchall()
        existing = result[0] == 1

        query = f"""
        SELECT COUNT(column_name)
        FROM information_schema.columns
        WHERE table_name = '{DataStore.PODCAST_EMBEDS}'
        AND column_name IN ('title', 'author', 'rating',
        'vec_cat', 'embedded_cat', 'vec_desc', 'embedded_desc')
        """
        result, *_ = conn.sql(query).fetchall()
        complete = result[0] == 7

        return existing and complete

    def _prep_categories(self, conn: duckdb.DuckDBPyConnection) -> None:
        dim = self.embedder.cat_model.config.hidden_size
        DataStore._logger.info('Creating categories vector store...')
        DataStore._with_transaction(conn, [
            f"""
            UPDATE {DataStore.CATEGORY_TAB}
            SET category = LOWER(category)""",
            f"DROP TABLE IF EXISTS {DataStore.CATEGORY_EMBEDS}",
            f"""
            CREATE TABLE {DataStore.CATEGORY_EMBEDS}(
                name VARCHAR PRIMARY KEY,
                vec FLOAT[{dim}]
                DEFAULT [0 for x in range({dim})]::FLOAT[{dim}])"""
        ])

    def _prep_reviews(self, conn: duckdb.DuckDBPyConnection) -> None:
        dim = self.embedder.rev_model.config.hidden_size
        DataStore._logger.info('Creating reviews vector store...')
        DataStore._with_transaction(conn, [
            f"""ALTER TABLE {DataStore.REVIEW_TAB}
            ADD COLUMN IF NOT EXISTS rev_id INTEGER""",
            f"""
            UPDATE {DataStore.REVIEW_TAB}
            SET rev_id = seq.seq_num
            FROM (
                SELECT ROW_NUMBER() OVER () AS seq_num, rowid
                FROM {DataStore.REVIEW_TAB}
            ) AS seq
            WHERE {DataStore.REVIEW_TAB}.rowid = seq.rowid""",
            f"""ALTER TABLE {DataStore.REVIEW_TAB}
            ADD COLUMN IF NOT EXISTS
            embedded BOOLEAN DEFAULT false""",
            f"""ALTER TABLE {DataStore.REVIEW_TAB}
            ADD COLUMN IF NOT EXISTS
            sentiment VARCHAR DEFAULT 'neutral'""",
            f"""ALTER TABLE {DataStore.REVIEW_TAB}
            ADD COLUMN IF NOT EXISTS vec FLOAT[{dim}]
            DEFAULT [0 for x in range({dim})]::FLOAT[{dim}]""",
            f"""
            UPDATE {DataStore.REVIEW_TAB} SET
                vec = [0 for x in range({dim})]::FLOAT[{dim}],
                embedded = false,
                sentiment = 'neutral'
            WHERE vec IS NULL
            """
        ])

    def _prep_episodes(self, conn: duckdb.DuckDBPyConnection) -> None:
        desc_dim = self.embedder.pod_model.config.hidden_size
        rev_dim = self.embedder.rev_model.config.hidden_size
        DataStore._logger.info('Creating episodes vector store...')
        DataStore._with_transaction(conn, [
            f"""
            ALTER TABLE {DataStore.PODCAST_TAB} ADD COLUMN
            IF NOT EXISTS ratings_count_value DOUBLE DEFAULT 0""",
            f"""
            ALTER TABLE {DataStore.PODCAST_TAB} ADD COLUMN
            IF NOT EXISTS embedded_desc BOOLEAN DEFAULT false""",
            f"""
            ALTER TABLE {DataStore.PODCAST_TAB} ADD COLUMN
            IF NOT EXISTS embedded_rev BOOLEAN DEFAULT false""",
            f"""
            ALTER TABLE {DataStore.PODCAST_TAB} ADD COLUMN
            IF NOT EXISTS vec_desc FLOAT[{desc_dim}]
            DEFAULT [0 for x in range({desc_dim})]::FLOAT[{desc_dim}]""",
            f"""
            ALTER TABLE {DataStore.PODCAST_TAB}
            ADD COLUMN IF NOT EXISTS vec_rev FLOAT[{rev_dim}]
            DEFAULT [0 for x in range({rev_dim})]::FLOAT[{rev_dim}]""",
            f"""
            UPDATE {DataStore.PODCAST_TAB}
            SET title = LOWER(title),
                author = LOWER(author),
                embedded_desc = false,
                embedded_rev = false,
                vec_desc = [0 for x in range({desc_dim})]::FLOAT[{desc_dim}],
                vec_rev = [0 for x in range({rev_dim})]::FLOAT[{rev_dim}],
                ratings_count_value = CASE
                    WHEN LOWER(SUBSTR(ratings_count,
                               LENGTH(ratings_count), 1)) = 'k'
                    THEN CAST(SUBSTR(ratings_count, 1,
                              LENGTH(ratings_count) - 1) AS DOUBLE) * 1000
                    WHEN LOWER(SUBSTR(ratings_count,
                               LENGTH(ratings_count), 1)) = 'm'
                    THEN CAST(SUBSTR(ratings_count, 1,
                              LENGTH(ratings_count) - 1) AS DOUBLE) * 1000000
                    WHEN LOWER(SUBSTR(ratings_count,
                               LENGTH(ratings_count), 1)) = 'b'
                    THEN CAST(SUBSTR(ratings_count, 1,
                              LENGTH(ratings_count) - 1) AS DOUBLE) * 1000000000
                    ELSE COALESCE(CAST(ratings_count AS DOUBLE), 0)
                END
            """
        ])

    def _prep_podcasts(self, conn: duckdb.DuckDBPyConnection) -> None:
        desc_dim = self.embedder.pod_model.config.hidden_size
        cat_dim = self.embedder.cat_model.config.hidden_size
        DataStore._logger.info('Creating podcasts vector store...')
        DataStore._with_transaction(conn, [
            f"DROP TABLE IF EXISTS {DataStore.PODCAST_EMBEDS}",
            f"""
            CREATE TABLE {DataStore.PODCAST_EMBEDS}(
                title VARCHAR,
                author VARCHAR,
                rating DOUBLE DEFAULT 0,
                vec_cat FLOAT[{cat_dim}]
                DEFAULT [0 for x in range({cat_dim})]::FLOAT[{cat_dim}],
                embedded_cat BOOLEAN DEFAULT false,
                vec_desc FLOAT[{desc_dim}]
                DEFAULT [0 for x in range({desc_dim})]::FLOAT[{desc_dim}],
                embedded_desc BOOLEAN DEFAULT false,
                PRIMARY KEY (title, author)
            )""",
            f"""
            INSERT INTO {DataStore.PODCAST_EMBEDS} (title, author, rating)
            SELECT title, author, COALESCE(
                SUM(average_rating * ratings_count_value)
                / NULLIF(SUM(ratings_count_value), 0), 0)
            FROM {DataStore.PODCAST_TAB}
            WHERE title IS NOT NULL AND author IS NOT NULL
            GROUP BY title, author"""
        ])

    @contextmanager
    def _conn(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        connection = duckdb.connect(self.file, config = {'threads': 1})
        connection.load_extension('vss')
        connection.execute("SET hnsw_enable_experimental_persistence = true")
        connection.execute("SET enable_progress_bar = true")

        try: yield connection
        except Exception as e:
            DataStore._logger.error(
                f'Could not obtain database connection curosr: {e}',
                exc_info=True
            )
        finally: connection.close()

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

    @staticmethod
    def _generate_updates(
            table: str,
            columns: tuple[str, ...],
            values: list[tuple[Any, ...]],
            condition_cols: tuple[str, ...],
            condition_values: list[tuple[Any, ...]]
    ) -> Generator[str, None, None]:
        for row_conditions, row_values in zip(condition_values, values):
            yield f"""UPDATE {table} SET
            {', '.join([f"{col} = {val}" for col, val in zip(columns, row_values)])}
            WHERE {
                ' AND '.join([
                    f"{col} = {val}"
                    for col, val in zip(condition_cols, row_conditions)
                ])
            }
            """
