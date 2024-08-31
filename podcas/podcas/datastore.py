from contextlib import contextmanager
from typing import Any, Generator, Optional
from os.path import abspath
from tqdm import tqdm
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
            embedder: Embedder
    ):
        self.file = abspath(path)
        self.embedder = embedder
        self._init_db()

    def get_reviews(
            self,
            top: int,
            rating_range: tuple[float, float],
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

    def get_podcasts(
            self,
            top: int,
            rating_range: tuple[float, float],
            title: Optional[str],
            fuzzy_title: bool,
            author: Optional[str],
            fuzzy_author: bool,
            category_embeddings: Optional[list[float]],
            review_embeddings: Optional[list[float]],
            desc_embeddings: Optional[list[float]],
            rating_boost: bool
    ) -> list[tuple[str, str, float, float]]:
        query = "SELECT title, author, rating,"

        if not (
                category_embeddings
                or review_embeddings
                or desc_embeddings
        ): query += ' 1'
        else:
            scores = []

            if category_embeddings:
                scores.append(f"""
                    (
                        1 + array_cosine_similarity(vec_cat,
                        {category_embeddings}::FLOAT[{len(category_embeddings)}])
                    ) / 2
                """)

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

            query += ' + '.join(scores)
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
        {"AND cat_embedded" if category_embeddings else ""}
        {"AND rev_embedded" if review_embeddings else ""}
        {"AND desc_embedded" if desc_embeddings else ""}
        ORDER BY score DESC, rating DESC
        {", levenshtein(title, '" + title.replace("'", "''") + "') ASC" if title else ""}
        {", levenshtein(author, '" + author.replace("'", "''") + "') ASC" if author else ""}
        LIMIT {top}
        """

        with self._conn() as conn:
            try: result = conn.sql(query).fetchall()
            except duckdb.ProgrammingError as error:
                DataStore._logger.error(
                    f"Input-caused error while getting podcasts: {error}",
                    exc_info=True
                )
                result = []

        return result

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
        FROM {DataStore.CATEGORY_TAB}
        WHERE category IS NOT NULL"""

        categories: list[str] = [
            result[0]
            for result in conn.sql(query).fetchall()
        ]

        category_embeddings = self.embedder.embed_categories(categories)
        dim = (
            len(category_embeddings[categories[0]])
            if len(categories) > 0 else Embedder.DEFAULT_VEC_SIZE
        )

        DataStore._logger.info('Creating categories vector store...')
        DataStore._with_transaction(conn, [
            f"DROP TABLE IF EXISTS {DataStore.CATEGORY_EMBEDS}",
            f"""
            CREATE TABLE {DataStore.CATEGORY_EMBEDS}(
                name VARCHAR PRIMARY KEY,
                vec FLOAT[{dim}])"""
        ])

        DataStore._logger.info('Ingesting categories embeddings...')
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
        reviews: list[tuple[str, str, str]] = conn.sql(f"""
        SELECT rev_id, title, content
        FROM {DataStore.REVIEW_TAB}
        WHERE title IS NOT NULL OR content IS NOT NULL
        """).fetchall()

        review_embeddings = self.embedder.embed_reviews(reviews)
        dim = (
            len(review_embeddings[0])
            if len(review_embeddings) > 0 else Embedder.DEFAULT_VEC_SIZE
        )

        DataStore._with_transaction(conn, [
            f"ALTER TABLE {DataStore.REVIEW_TAB} DROP COLUMN IF EXISTS vec",
            f"""ALTER TABLE {DataStore.REVIEW_TAB}
            ADD COLUMN vec FLOAT[{dim}]
            DEFAULT [0 for x in range({dim})]::FLOAT[{dim}]""",
            f"""ALTER TABLE {DataStore.REVIEW_TAB}
            ADD COLUMN embedded BOOLEAN DEFAULT false"""
        ])

        DataStore._logger.info('Ingesting reviews embeddings...')
        ids = [(idx,) for idx, _, _ in reviews]
        for update in tqdm(
            DataStore._generate_updates(
                DataStore.REVIEW_TAB,
                ('vec', 'embedded'),
                [(embed, 'true') for embed in review_embeddings],
                ('rev_id',),
                ids
            ),
            total=len(ids),
            desc="Ingesting reviews embeddings"
        ): conn.execute(update)

        DataStore._logger.info('Indexing reviews vector space...')
        conn.sql(f"""
        CREATE INDEX idx_rev
        ON {DataStore.REVIEW_TAB} USING HNSW (vec)
        WITH (metric = 'cosine')""")

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

        category_rows = conn.execute(f"""
        SELECT
            p.title, p.author,
            LIST(c.category) AS categories
        FROM {DataStore.PODCAST_TAB} p
        JOIN {DataStore.CATEGORY_TAB} c
        ON p.podcast_id = c.podcast_id
        WHERE p.title IS NOT NULL AND p.author IS NOT NULL
        GROUP BY p.author, p.title""").fetchall()

        desc_embeds, rev_embeds, cat_embeds = self.embedder.embed_podcasts(
            [descriptions for _, _, descriptions in desc_rows],
            [reviews for _, _, reviews in review_rows],
            [categories for _, _, categories in category_rows]
        )

        desc_dim, rev_dim, cat_dim = (
            len(desc_embeds[0]) if len(desc_embeds) > 0 else Embedder.DEFAULT_VEC_SIZE,
            len(rev_embeds[0]) if len(rev_embeds) > 0 else Embedder.DEFAULT_VEC_SIZE,
            len(cat_embeds[0]) if len(cat_embeds) > 0 else Embedder.DEFAULT_VEC_SIZE
        )

        DataStore._logger.info('Creating podcasts vector store...')
        DataStore._with_transaction(conn, [
            f"DROP TABLE IF EXISTS {DataStore.PODCAST_EMBEDS}",
            f"""
            CREATE TABLE {DataStore.PODCAST_EMBEDS}(
                title VARCHAR,
                author VARCHAR,
                rating DOUBLE DEFAULT 0,
                vec_desc FLOAT[{desc_dim}]
                DEFAULT [0 for x in range({desc_dim})]::FLOAT[{desc_dim}],
                desc_embedded BOOLEAN DEFAULT false,
                vec_rev FLOAT[{rev_dim}]
                DEFAULT [0 for x in range({rev_dim})]::FLOAT[{rev_dim}],
                rev_embedded BOOLEAN DEFAULT false,
                vec_cat FLOAT[{cat_dim}]
                DEFAULT [0 for x in range({cat_dim})]::FLOAT[{cat_dim}],
                cat_embedded BOOLEAN DEFAULT false
            )""",
            f"""
            INSERT INTO {DataStore.PODCAST_EMBEDS} (title, author, rating)
            SELECT title, author, COALESCE(
                SUM(average_rating * ratings_count_value)
                / NULLIF(SUM(ratings_count_value), 0), 0)
            FROM {DataStore.PODCAST_TAB}
            WHERE title IS NOT NULL OR author IS NOT NULL
            GROUP BY title, author"""
        ])

        title_author_desc_couples = [
            (
                f"""'{title.replace("'", "''")}'""",
                f"""'{author.replace("'", "''")}'""")
            for title, author, _ in desc_rows
        ]
        for update in tqdm(
            DataStore._generate_updates(
                DataStore.PODCAST_EMBEDS,
                ('vec_desc', 'desc_embedded'),
                [(embed, 'true') for embed in desc_embeds],
                ('title', 'author'),
                title_author_desc_couples
            ),
            total=len(title_author_desc_couples),
            desc = "Ingesting podcasts description embeddings"
        ): conn.execute(update)

        title_author_rev_couples = [
            (
                f"""'{title.replace("'", "''")}'""",
                f"""'{author.replace("'", "''")}'""")
            for title, author, _ in review_rows
        ]
        for update in tqdm(
            DataStore._generate_updates(
                DataStore.PODCAST_EMBEDS,
                ('vec_rev', 'rev_embedded'),
                [(embed, 'true') for embed in rev_embeds],
                ('title', 'author'),
                title_author_rev_couples
            ),
            total = len(title_author_rev_couples),
            desc = "Ingesting podcasts review embeddings"
        ): conn.execute(update)

        title_author_cat_couples = [
            (
                f"""'{title.replace("'", "''")}'""",
                f"""'{author.replace("'", "''")}'""")
            for title, author, _ in category_rows
        ]
        for update in tqdm(
            DataStore._generate_updates(
                DataStore.PODCAST_EMBEDS,
                ('vec_cat', 'cat_embedded'),
                [(embed, 'true') for embed in cat_embeds],
                ('title', 'author'),
                title_author_cat_couples
            ),
            total = len(title_author_cat_couples),
            desc = "Ingesting podcasts category embeddings"
        ): conn.execute(update)

        DataStore._logger.info('Indexing podcasts vector space...')
        DataStore._with_transaction(conn, [
            f"""
            CREATE INDEX idx_pod_desc
            ON {DataStore.PODCAST_EMBEDS} USING HNSW (vec_desc)
            WITH (metric = 'cosine')""",
            f"""
            CREATE INDEX idx_pod_rev
            ON {DataStore.PODCAST_EMBEDS} USING HNSW (vec_rev)
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
            ALTER TABLE {DataStore.PODCAST_TAB}
            DROP COLUMN IF EXISTS ratings_count_value""",
            f"""
            ALTER TABLE {DataStore.PODCAST_TAB}
            ADD COLUMN ratings_count_value DOUBLE DEFAULT 0""",
            f"""
            UPDATE {DataStore.PODCAST_TAB}
            SET title = LOWER(title),
                author = LOWER(author),
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
                    ELSE CAST(ratings_count AS DOUBLE)
                END
            """
        ])

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
