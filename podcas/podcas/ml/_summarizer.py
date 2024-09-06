from logging import getLogger
from transformers import pipeline

from podcas import lib_device


class Summarizer:
    """
    A class for summarizing text using a pre-trained summarization model.

    Attributes:
        TASK_NAME: The task name used by the transformers pipeline.
        model_name: Name or path of the summarization model.
        max_length: Maximum length of the summary.
        batch_size: Batch size for summarization processing.
        _summarizer: The summarization pipeline object.
    """

    TASK_NAME = "summarization"
    _logger = getLogger(f"{__name__}.{__qualname__}")

    def __init__(
            self,
            model: str,
            max_length: int = 64,
            batch_size: int = 8
    ) -> None:
        """
        Initializes the Summarizer with a specified model and configuration.

        Args:
            model: The model name or path used for the summarization pipeline.
            max_length: Maximum token length for the summaries.
            batch_size: Number of texts to summarize per batch.
        """
        self.model_name = model
        self.max_length = max_length
        self.batch_size = batch_size

        self._summarizer = pipeline(
            Summarizer.TASK_NAME,
            model = model,
            device = lib_device
        )

    def summarize_inplace(self, inputs: list[str], max_tries: int = 2) -> None:
        """
        Recursively summarizes texts in place while their tokens represent
        a longer context than 1.5 times the truncation window.

        Args:
            inputs: List of texts to summarize in place.
            max_tries: Maximum number of recursive attempts to shorten the texts.

        Raises:
            ValueError: If the summarization pipeline fails during tokenization
                        or summarization.
        """

        Summarizer._logger.info("Summarizing input...")
        if (
                not inputs
                or not max_tries
                or not self._summarizer.tokenizer
        ): return

        tokens = self._summarizer.tokenizer(inputs)['input_ids']
        if tokens is None or not isinstance(tokens, list):
            error_msg = "Summarization pipeline failed - @ tokenization!"
            Summarizer._logger.error(error_msg)
            raise ValueError(error_msg)

        to_summarize_idx = [
            idx
            for idx, idx_tokens in enumerate(tokens)
            if len(idx_tokens) > (3 * self.max_length / 2 - 2)
        ]

        if not to_summarize_idx: return

        summarized = self._summarizer(
            [inputs[idx] for idx in to_summarize_idx],
            batch_size = self.batch_size,
            do_sample = False
        )

        if (
                summarized is None
                or not isinstance(summarized, list)
                or not isinstance(summarized[0], dict)
                or 'summary_text' not in summarized[0]
        ):
            error_msg = "Summarization pipeline failed - @ summarization!"
            Summarizer._logger.error(error_msg)
            raise ValueError(error_msg)

        for idx, summary in zip(to_summarize_idx, summarized):
            inputs[idx] = summary['summary_text']

        self.summarize_inplace(inputs, max_tries - 1)
