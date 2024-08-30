from threading import Lock
from logging import getLogger
from typing import Optional, Self

from .datastore import DataStore
from .embedder import Embedder

class PodcastSearch:
    __instance = None
    __lock = Lock()
    _logger = getLogger(f"{__name__}.{__qualname__}")

    def __new__(cls, *args, **kwargs):
        with cls.__lock:
            if not cls.__instance:
                cls.__instance = super(PodcastSearch, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        self._top = 3
        self._min = 0
        self._max = 5
        self._rating_boosted = False
        self._title: Optional[str] = None
        self._author: Optional[str] = None
        self._category_emb: Optional[list[float]] = None
        self._review_emb: Optional[list[float]] = None
        self._desc_emb: Optional[list[float]] = None
        self.__embedder = Embedder(
            category_model = 'all-MiniLM-L6-v2',
            review_model = 'multi-qa-MiniLM-L6-cos-v1',
            podcast_model = 'multi-qa-MiniLM-L6-cos-v1'
        )

    def load(self, *, source: str) -> Self:
        self.source = source
        self.__db = DataStore(self.source, self.__embedder)
        return self

    def using(
            self, *,
            category_model: str = 'all-MiniLM-L6-v2',
            review_model: str = 'multi-qa-MiniLM-L6-cos-v1',
            podcast_model: str = 'multi-qa-MiniLM-L6-cos-v1'
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

    def by_rating(self, min: float, max: float = 5.) -> Self:
        self._min = min
        self._max = max
        return self

    def by_title(self, title: str) -> Self:
        self.title = title
        return self

    def by_author(self, author: str) -> Self:
        self.author = author
        return self

    def by_category(self, category: str) -> Self:
        PodcastSearch._logger.info("Embedding query category...")
        embeddings = self.__embedder.cat_embedder.encode(category)
        self._category_emb = embeddings[0].tolist()
        return self

    def by_review(self, review: str) -> Self:
        PodcastSearch._logger.info("Embedding query review...")
        embeddings = self.__embedder.rev_embedder.encode(review)
        self._review_emb = embeddings[0].tolist()
        return self

    def by_description(self, query: str) -> Self:
        PodcastSearch._logger.info("Embedding query description...")
        embeddings = self.__embedder.pod_embedder.encode(query)
        self._desc_emb = embeddings[0].tolist()
        return self

    def get(self) -> list[tuple[str, str, float, float]]:
        """
        If different kinds of embeddings are set
        then the query search is associative.
        Example:
            [...].by_category("fiction")
                 .by_description("monsters")
        will end up searching podcasts that match category
        "fiction" OR description "monsters"
        """
        PodcastSearch._logger.info("Executing query...")
        podcasts = self.__db.get_podcasts(
            self._top,
            (self._min, self._max),
            self._title,
            self._author,
            self._category_emb,
            self._review_emb,
            self._desc_emb,
            self._rating_boosted
        )

        self._top = 3
        self._min = 0
        self._max = 5
        self._rating_boosted = False
        self._title = None
        self._author = None
        self._category_emb = None
        self._review_emb = None
        self._desc_emb = None

        return podcasts
