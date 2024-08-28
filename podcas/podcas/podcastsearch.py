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
            path: str,
            category_model: str = 'all-MiniLM-L6-v2',
            review_model: str = 'distiluse-base-multilingual-cased-v1',
            podcast_model: str = 'distiluse-base-multilingual-cased-v1'
    ):
        self.source = path
        self.embedder = Embedder(
            category_model = category_model,
            review_model = review_model,
            podcast_model = podcast_model
        )
        self._db = DataStore(path, self.embedder)
