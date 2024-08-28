from __future__ import annotations
from threading import Lock
from logging import getLogger

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

    def __init__(
            self,
            path: str, *,
            category_model: str = 'all-MiniLM-L6-v2',
            review_model: str = 'distiluse-base-multilingual-cased-v1',
            podcast_model: str = 'distiluse-base-multilingual-cased-v1',
            # TODO
            rank_boost: bool = False
    ):
        self.source = path
        self._embedder = Embedder(
            category_model = category_model,
            review_model = review_model,
            podcast_model = podcast_model
        )
        self._db = DataStore(path, self._embedder)

    # TODO
    def by_score(self, min: float, max: float = 5.) -> PodcastSearch: ...
    def by_title(self, title: str) -> PodcastSearch: ...
    def by_author(self, author: str) -> PodcastSearch: ...
    def by_category(self, category: str) -> PodcastSearch: ...
    def by_review(self, review: str) -> PodcastSearch: ...
    def by_query(self, query: str) -> PodcastSearch: ...
