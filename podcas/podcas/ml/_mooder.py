from logging import getLogger
from typing import Optional
from transformers import pipeline

from podcas import lib_device
from podcas.ml import Summarizer

class Mooder:
    TASK_NAME = "sentiment-analysis"
    _logger = getLogger(f"{__name__}.{__qualname__}")

    def __init__(
            self,
            model: str,
            summarizer: Optional[Summarizer] = None,
            max_length: int = 256,
            batch_size: int = 32
    ) -> None:
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
        Mooder._logger.info('Analyzing reviews...')
        aggregated = [
            f'{title}:{content}'
            for _, title, content in reviews
        ]

        try: return self.analyze(aggregated)
        except: return ['' for _ in aggregated]
