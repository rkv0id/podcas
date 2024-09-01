from logging import getLogger
from transformers import pipeline

from podcas import lib_device


class Summarizer:
    TASK_NAME = "summarization"
    _logger = getLogger(f"{__name__}.{__qualname__}")

    def __init__(
            self,
            model: str,
            max_length: int = 512,
            batch_size: int = 32
    ) -> None:
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
        Recursively summarizes text while its tokens represent
        a longer context than 1.5 times the truncation window
        """

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
        to_summarize = [inputs[idx] for idx in to_summarize_idx]

        summarized = self._summarizer(
            to_summarize,
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
