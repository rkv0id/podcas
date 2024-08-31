from threading import Lock
from logging import getLogger
from typing import Optional, Self

from .datastore import DataStore
from .embedder import Embedder


class ReviewSearch:
    DEFAULT_EMBEDDING_MODEL = "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking"
    __instance = None
    __lock = Lock()
    _logger = getLogger(f"{__name__}.{__qualname__}")

    def __new__(cls, *args, **kwargs):
        with cls.__lock:
            if not cls.__instance:
                cls.__instance = super(ReviewSearch, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        self._top = 3
        self._min = 0
        self._max = 5
        self._rating_boosted = False
        self._query_emb: Optional[list[float]] = None
        self.__embedder = Embedder(
            category_model = ReviewSearch.DEFAULT_EMBEDDING_MODEL,
            review_model = ReviewSearch.DEFAULT_EMBEDDING_MODEL,
            podcast_model = ReviewSearch.DEFAULT_EMBEDDING_MODEL
        )

    def load(self, *, source: str) -> Self:
        self.source = source
        self.__db = DataStore(self.source, self.__embedder)
        return self

    def using(
            self, *,
            category_model: str = 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking',
            review_model: str = 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking',
            podcast_model: str = 'sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking'
    ):
        self.__embedder = Embedder(
            category_model = category_model,
            review_model = review_model,
            podcast_model = podcast_model
        )
        return self

    def top(self, n: int) -> Self:
        self._top = n
        return self

    def rating_boosted(self, boost: bool = True) -> Self:
        self._rating_boosted = boost
        return self

    def by_rating(self, min: float, max: float = 5.) -> Self:
        self._min = min
        self._max = max
        return self

    def by_query(self, query: str) -> Self:
        ReviewSearch._logger.info("Embedding query...")
        embeddings = Embedder.embed_text(
            [query],
            self.__embedder.rev_tokenizer,
            self.__embedder.rev_model
        )

        self._query_emb = embeddings[0].tolist()
        return self

    def boost_by_rank(self, boost: bool = True) -> Self:
        self._rating_boosted = boost
        return self

    def get(self) -> list[tuple[str, str, float, float]]:
        ReviewSearch._logger.info("Executing query...")
        reviews = self.__db.get_reviews(
            self._top,
            (self._min, self._max),
            self._query_emb,
            self._rating_boosted
        )

        self._top = 3
        self._min = 0
        self._max = 5
        self._rating_boosted = False
        self._query_emb = None

        return reviews
