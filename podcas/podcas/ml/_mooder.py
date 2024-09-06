from logging import getLogger
from typing import Optional
from transformers import pipeline

from podcas import lib_device
from podcas.ml import Summarizer

class Mooder:
    """
    A class for analyzing the sentiment of text inputs using a pre-trained sentiment analysis model.

    Attributes:
        TASK_NAME: The task name used by the transformers pipeline.
        model_name: The name or path of the sentiment analysis model.
        max_length: Maximum sequence length for the sentiment analysis.
        batch_size: Batch size for processing inputs.
        _analyzer: The sentiment analysis pipeline object.
        summarizer: Optional summarizer for text preprocessing.
        labels: A set of sentiment labels supported by the model.
    """

    TASK_NAME = "sentiment-analysis"
    _logger = getLogger(f"{__name__}.{__qualname__}")

    def __init__(
            self,
            model: str,
            summarizer: Optional[Summarizer] = None,
            max_length: int = 256,
            batch_size: int = 32
    ) -> None:
        """
        Initializes the Mooder with a specified model, summarizer, and configuration.

        Args:
            model: The name or path of the sentiment analysis model.
            summarizer: An optional Summarizer object for preprocessing input texts.
            max_length: Maximum token length for inputs to the sentiment analysis model.
            batch_size: Number of texts to process per batch.
        """
        self.model_name = model
        self.max_length = max_length
        self.batch_size = batch_size

        self._analyzer = pipeline(
            Mooder.TASK_NAME,
            model=model,
            device=lib_device
        )

        self.summarizer = summarizer

        model_label2id = self._analyzer.model.config.label2id
        self.labels = set(
            [key.lower() for key in model_label2id.keys()]
            if model_label2id
            else ["positive", "negative"]
        )

    def analyze(self, inputs: list[str]) -> list[str]:
        """
        Analyzes the sentiment of a list of input texts.

        Args:
            inputs: A list of texts to analyze.

        Returns:
            A list of sentiment labels corresponding to each input text.

        Raises:
            ValueError: If the sentiment analysis pipeline fails.
        """
        if self.summarizer:
            summarized = inputs.copy()
            try: self.summarizer.summarize_inplace(summarized)
            except Exception as e:
                Mooder._logger.error(
                    f"Error occured while summarizing: {e}",
                    exc_info=True
                )
                raise e
        else: summarized = inputs

        results = self._analyzer(
            summarized,
            batch_size = self.batch_size,
            max_length = self.max_length,
            truncation = True,
            padding = True
        )

        if results is None or not isinstance(results, list):
            error_msg = "Sentiment analysis pipeline failed!"
            Mooder._logger.error(error_msg)
            raise ValueError(error_msg)

        return [each['label'] for each in results]

    def analyze_reviews(
            self,
            reviews: list[tuple[int, str, str]]
    ) -> list[str]:
        """
        Analyzes the sentiment of a list of reviews.

        Args:
            reviews: A list of reviews, each represented as a tuple (id, title, content).

        Returns:
            A list of sentiment labels corresponding to each review.
        """
        Mooder._logger.info('Analyzing reviews...')
        aggregated = [
            f'{title}:{content}'
            for _, title, content in reviews
        ]

        try: return self.analyze(aggregated)
        except: return ['' for _ in aggregated]
