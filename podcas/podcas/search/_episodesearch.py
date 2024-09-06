from threading import Lock
from logging import getLogger
from typing import Optional, Self

from podcas.data import DataStore, Episode
from podcas.ml import Embedder, Mooder, Summarizer
from podcas import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SENTIMENT_MODEL,
    DEFAULT_SUMMARIZE_MODEL
)


class EpisodeSearch:
    """
    Singleton class for managing podcast episode searches with embedded text and other filters.

    Attributes:
        __instance: A class-level singleton instance of EpisodeSearch.
        __lock: A threading lock to ensure thread-safe singleton creation.
        _top: Number of top episodes to retrieve.
        _min: Minimum rating for filtering episodes.
        _max: Maximum rating for filtering episodes.
        _rating_boosted: Boolean flag indicating if rating boost is applied.
        _title: Title filter for episodes.
        _author: Author filter for episodes.
        _fuzzy_title: Boolean flag indicating if fuzzy matching should be applied to the title.
        _fuzzy_author: Boolean flag indicating if fuzzy matching should be applied to the author.
        _rev_emb: Embedded review vector for similarity matching.
        _desc_emb: Embedded description vector for similarity matching.
        __summarizer: Summarizer for text summarization (optional).
        __embedder: Embedder for generating text embeddings.
        __mooder: Mooder for sentiment analysis.
        __db: DataStore object for managing episode data.

    Methods:
        load: Initializes the DataStore with the given data source.
        using: Configures models for embedding, sentiment analysis, and summarization.
        top: Sets the number of top episodes to retrieve.
        rating_boosted: Sets the rating boost flag.
        by_rating: Filters episodes by rating range.
        by_title: Filters episodes by title, with optional fuzzy matching.
        by_author: Filters episodes by author, with optional fuzzy matching.
        by_review: Embeds the review text for similarity-based retrieval.
        by_description: Embeds the description text for similarity-based retrieval.
        get: Executes the search and returns the filtered episodes.
    """

    __instance = None
    __lock = Lock()
    _logger = getLogger(f"{__name__}.{__qualname__}")

    def __new__(cls, *args, **kwargs):
        with cls.__lock:
            if not cls.__instance:
                cls.__instance = super(EpisodeSearch, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        """
        Initializes the EpisodeSearch instance with default configurations.
        """
        self._top = 3
        self._min = 0
        self._max = 5
        self._rating_boosted = False
        self._title: Optional[str] = None
        self._author: Optional[str] = None
        self._fuzzy_title: bool = False
        self._fuzzy_author: bool = False
        self._rev_emb: Optional[list[float]] = None
        self._desc_emb: Optional[list[float]] = None
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
        Loads the data source for the episode search.

        Args:
            source: The path or URL of the data source.

        Returns:
            Self: Returns the EpisodeSearch instance.
        """
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
        """
        Configures the models used for embedding, sentiment analysis, and summarization.

        Args:
            category_model: The model name or path for category embedding.
            review_model: The model name or path for review embedding.
            podcast_model: The model name or path for podcast embedding.
            mooder_model: The model name or path for sentiment analysis.
            summary_model: Optional model name or path for summarization.

        Returns:
            Self: Returns the EpisodeSearch instance.
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
        Sets the number of top episodes to retrieve.

        Args:
            n: The number of top episodes.

        Returns:
            Self: Returns the EpisodeSearch instance.
        """
        self._top = n
        return self

    def rating_boosted(self, boost: bool = True) -> Self:
        """
        Sets the rating boost flag. Boosting biases
        the search towards higher-rated episodes.

        Args:
            boost: Boolean flag to enable or disable rating boost.

        Returns:
            Self: Returns the EpisodeSearch instance.
        """
        self._rating_boosted = boost
        return self

    def by_rating(self, min: float, max: float = 5.) -> Self:
        """
        Filters episodes by a specified rating range.

        Args:
            min: Minimum rating for filtering.
            max: Maximum rating for filtering.

        Returns:
            Self: Returns the EpisodeSearch instance.
        """
        self._min = min
        self._max = max
        return self

    def by_title(self, title: str, *, fuzzy: bool = False) -> Self:
        """
        Filters episodes by title, with optional fuzzy matching.

        Args:
            title: Title to filter by.
            fuzzy: Boolean flag for fuzzy matching.

        Returns:
            Self: Returns the EpisodeSearch instance.
        """
        self._title = title
        self._fuzzy_title = fuzzy
        return self

    def by_author(self, author: str, *, fuzzy: bool = False) -> Self:
        """
        Filters episodes by author, with optional fuzzy matching.

        Args:
            author: Author to filter by.
            fuzzy: Boolean flag for fuzzy matching.

        Returns:
            Self: Returns the EpisodeSearch instance.
        """
        self._author = author
        self._fuzzy_author = fuzzy
        return self

    def by_review(self, review: str) -> Self:
        """
        Embeds the review text for similarity-based retrieval.

        Args:
            review: The review text to embed.

        Returns:
            Self: Returns the EpisodeSearch instance.
        """
        EpisodeSearch._logger.info("Embedding review query...")
        embeddings = self.__embedder.embed_text(
            [review],
            self.__embedder.rev_tokenizer,
            self.__embedder.rev_model
        )

        self._rev_emb = embeddings[0].tolist()
        return self

    def by_description(self, query: str) -> Self:
        """
        Embeds the description text for similarity-based retrieval.

        Args:
            query: The description text to embed.

        Returns:
            Self: Returns the EpisodeSearch instance.
        """
        EpisodeSearch._logger.info("Embedding description query...")
        embeddings = self.__embedder.embed_text(
            [query],
            self.__embedder.pod_tokenizer,
            self.__embedder.pod_model
        )

        self._desc_emb = embeddings[0].tolist()
        return self

    def get(self) -> list[Episode]:
        """
        Executes the search and returns the filtered podcast episodes.

        The search is associative if different embeddings
        are set for different attributes. For example:
            [...].by_category("fiction")
                 .by_description("monsters")
        will search for episodes that match either the category "fiction" OR description "monsters".

        Returns:
            A list of tuples containing episode data (
                title,
                author,
                itunes_id,
                rating,
                similarity score
            ).
        """
        EpisodeSearch._logger.info("Executing query...")
        episodes = self.__db.get_episodes(
            self._top,
            (self._min, self._max),
            self._title,
            self._fuzzy_title,
            self._author,
            self._fuzzy_author,
            self._rev_emb,
            self._desc_emb,
            self._rating_boosted
        )

        self._top = 3
        self._min = 0
        self._max = 5
        self._rating_boosted = False
        self._title = None
        self._fuzzy_title = False
        self._author = None
        self._fuzzy_author = False
        self._rev_emb = None
        self._desc_emb = None

        return episodes
