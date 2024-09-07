from logging import getLogger
from typing import Optional, Self

from podcas.data import DataStore, Review
from podcas.ml import Embedder, Mooder, Summarizer
from podcas import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SENTIMENT_MODEL,
    DEFAULT_SUMMARIZE_MODEL
)


class ReviewSearch:
    """
    Singleton class for managing review searches with embedded text and sentiment analysis.

    Attributes:
        __instance: A class-level singleton instance of ReviewSearch.
        _top: Number of top reviews to retrieve.
        _min: Minimum rating for filtering reviews.
        _max: Maximum rating for filtering reviews.
        _rating_boosted: Boolean flag indicating if rating boost is applied.
        _sentiment: Desired sentiment filter (positive, negative, or neutral).
        _query_emb: Embedded query vector for similarity matching.
        __summarizer: Summarizer for text summarization (optional).
        __embedder: Embedder for generating text embeddings.
        __mooder: Mooder for sentiment analysis.
        _db: DataStore object for managing review data.

    Methods:
        load: Initializes the DataStore with the given data source.
        using: Configures models for embedding, sentiment analysis, and summarization.
        top: Sets the number of top reviews to retrieve.
        rating_boosted: Sets the rating boost flag.
        by_rating: Filters reviews by rating range.
        positive: Filters reviews by positive sentiment.
        negative: Filters reviews by negative sentiment.
        neutral: Filters reviews by neutral sentiment.
        by_query: Embeds the query text for similarity-based retrieval.
        get: Executes the search and returns the filtered reviews.
    """

    __instance = None
    _logger = getLogger(f"{__name__}.{__qualname__}")

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(ReviewSearch, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        """
        Initializes the ReviewSearch instance with default configurations.
        """
        self._db = None
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
        """
        Loads the data source for the review search.

        Args:
            source: The path or URL of the data source.

        Returns:
            Self: Returns the ReviewSearch instance.
        """
        self.source = source
        self._db = DataStore(self.source, self.__embedder, self.__mooder)
        return self

    def using(
            self, *,
            category_model: str = DEFAULT_EMBEDDING_MODEL,
            review_model: str = DEFAULT_EMBEDDING_MODEL,
            podcast_model: str = DEFAULT_EMBEDDING_MODEL,
            mooder_model: str = DEFAULT_SENTIMENT_MODEL,
            summary_model: Optional[str] = None
    ):
        """
        Configures the models used for embedding, sentiment analysis, and summarization.

        Args:
            category_model: The model name or path for category embedding.
            review_model: The model name or path for review embedding.
            podcast_model: The model name or path for podcast embedding.
            mooder_model: The model name or path for sentiment analysis.
            summary_model: Optional model name or path for summarization.

        Returns:
            Self: Returns the ReviewSearch instance.
        """
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
        """
        Sets the number of top reviews to retrieve.

        Args:
            n: The number of top reviews.

        Returns:
            Self: Returns the ReviewSearch instance.
        """
        self._top = n
        return self

    def rating_boosted(self, boost: bool = True) -> Self:
        """
        Sets the rating boost flag. Boosting biases
        the search towards higher-rated reviews.

        Args:
            boost: Boolean flag to enable or disable rating boost.

        Returns:
            Self: Returns the ReviewSearch instance.
        """
        self._rating_boosted = boost
        return self

    def by_rating(self, min: float, max: float = 5.) -> Self:
        """
        Filters reviews by a specified rating range.

        Args:
            min: Minimum rating for filtering.
            max: Maximum rating for filtering.

        Returns:
            Self: Returns the ReviewSearch instance.
        """
        self._min = min
        self._max = max
        return self

    def positive(self) -> Self:
        """
        Filters reviews by positive sentiment.

        Returns:
            Self: Returns the ReviewSearch instance.
        """
        self._sentiment = 'positive'
        return self

    def negative(self) -> Self:
        """
        Filters reviews by positive sentiment.

        Returns:
            Self: Returns the ReviewSearch instance.
        """
        self._sentiment = 'negative'
        return self

    def neutral(self) -> Self:
        """
        Filters reviews by neutral sentiment.
        (if applicable according to model used)

        Returns:
            Self: Returns the ReviewSearch instance.
        """
        self._sentiment = 'neutral'
        return self

    def by_query(self, query: str) -> Self:
        """
        Embeds the query text for similarity-based retrieval.

        Args:
            query: The query text to embed.

        Returns:
            Self: Returns the ReviewSearch instance.
        """
        ReviewSearch._logger.info("Embedding query...")
        embeddings = self.__embedder.embed_text(
            [query],
            self.__embedder.rev_tokenizer,
            self.__embedder.rev_model
        )

        self._query_emb = embeddings[0].tolist()
        return self

    def get(self) -> list[Review]:
        """
        Executes the search and returns the filtered reviews.

        Returns:
            A list of tuples containing review data (title, content, rating, similarity score).
        """
        ReviewSearch._logger.info("Executing query...")
        if self._db:
            reviews = self._db.get_reviews(
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
        else:
            error = "Cannot fetch results. Datastore not initialised!"
            ReviewSearch._logger.error(error)
            raise ValueError(error)
