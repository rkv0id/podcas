from threading import Lock
from logging import getLogger
from typing import Optional, Self

from .datastore import DataStore
from .embedder import Embedder

class ReviewSearch:
    __instance = None
    __lock = Lock()
    _logger = getLogger(f"{__name__}.{__qualname__}")

    def __new__(cls, *args, **kwargs):
        with cls.__lock:
            if not cls.__instance:
                cls.__instance = super(ReviewSearch, cls).__new__(cls)
        return cls.__instance

    def __init__(
            self,
            path: str, *,
            category_model: str = 'all-MiniLM-L6-v2',
            review_model: str = 'multi-qa-MiniLM-L6-cos-v1',
            podcast_model: str = 'multi-qa-MiniLM-L6-cos-v1'
    ):
        self.source = path
        self.__embedder = Embedder(
            category_model = category_model,
            review_model = review_model,
            podcast_model = podcast_model
        )
        self.__db = DataStore(path, self.__embedder)
        self._top = 3
        self._min = 0
        self._max = 5
        self._rating_boosted = False
        self._query_emb: Optional[list[float]] = None

    def top(self, n: int) -> Self:
        self._top = n
        return self

    def by_rating(self, min: float, max: float = 5.) -> Self:
        self._min = min
        self._max = max
        return self

    def by_query(self, query: str) -> Self:
        embeddings = self.__embedder.rev_embedder.encode(query)
        self._query_emb = embeddings[0].tolist()
        return self

    def boost_by_rank(self, boost: bool = True) -> Self:
        self._rating_boosted = boost
        return self

    def get(self) -> list[tuple[str, str, float]]:
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
