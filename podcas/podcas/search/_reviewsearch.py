from threading import Lock
from logging import getLogger
from typing import Optional, Self

from podcas.data import DataStore
from podcas.ml import Embedder, Mooder, Summarizer
from podcas import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SENTIMENT_MODEL,
    DEFAULT_SUMMARIZE_MODEL
)


class ReviewSearch:
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
        self._sentiment: Optional[str] = None
        self._query_emb: Optional[list[float]] = None
        # GPU-POOR so no summarization for me :shrug:
        # self.__summarizer = Summarizer(DEFAULT_SUMMARIZE_MODEL)
        self.__summarizer: Optional[Summarizer] = None
        self.__embedder = Embedder(
            category_model = DEFAULT_EMBEDDING_MODEL,
            review_model = DEFAULT_EMBEDDING_MODEL,
            podcast_model = DEFAULT_EMBEDDING_MODEL,
            summarizer = self.__summarizer
        )
        self.__mooder = Mooder(
            model = DEFAULT_SENTIMENT_MODEL,
            summarizer = self.__summarizer
        )

    def load(self, *, source: str) -> Self:
        self.source = source
        self.__db = DataStore(self.source, self.__embedder, self.__mooder)
        return self

    def using(
            self, *,
            category_model: str = DEFAULT_EMBEDDING_MODEL,
            review_model: str = DEFAULT_EMBEDDING_MODEL,
            podcast_model: str = DEFAULT_EMBEDDING_MODEL,
            mooder_model: str = DEFAULT_SENTIMENT_MODEL,
            summary_model: Optional[str] = None
    ):
        self.__summarizer = (
            Summarizer(summary_model)
            if summary_model else None
        )
        self.__embedder = Embedder(
            category_model = category_model,
            review_model = review_model,
            podcast_model = podcast_model,
            summarizer = self.__summarizer
        )
        self.__mooder = Mooder(
            model = mooder_model,
            summarizer = self.__summarizer
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

    def positive(self) -> Self:
        self._sentiment = 'positive'
        return self

    def negative(self) -> Self:
        self._sentiment = 'negative'
        return self

    def neutral(self) -> Self:
        self._sentiment = 'neutral'
        return self

    def by_query(self, query: str) -> Self:
        ReviewSearch._logger.info("Embedding query...")
        embeddings = self.__embedder.embed_text(
            [query],
            self.__embedder.rev_tokenizer,
            self.__embedder.rev_model
        )

        self._query_emb = embeddings[0].tolist()
        return self

    def get(self) -> list[tuple[str, str, float, float]]:
        ReviewSearch._logger.info("Executing query...")
        reviews = self.__db.get_reviews(
            self._top,
            (self._min, self._max),
            self._sentiment,
            self._query_emb,
            self._rating_boosted
        )

        self._top = 3
        self._min = 0
        self._max = 5
        self._rating_boosted = False
        self._sentiment = None
        self._query_emb = None

        return reviews
