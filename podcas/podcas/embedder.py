from logging import getLogger
from sentence_transformers import SentenceTransformer


class Embedder:
    _logger = getLogger(f"{__name__}.{__qualname__}")

    def __init__(
            self,
            category_model: str,
            review_model: str,
            podcast_model: str
    ):
        self.category_embedder = SentenceTransformer(category_model)
        self.review_embedder = SentenceTransformer(review_model)
        self.podcast_embedder = SentenceTransformer(podcast_model)
        self.models = {
            'category': category_model,
            'review': review_model,
            'podcast': podcast_model
        }

    def embed_categories(self, categories: list[str]) -> dict[str, list[float]]:
        Embedder._logger.info("Embedding categories...")
        embeddings = self.category_embedder.encode(
            categories,
            show_progress_bar=True
        )

        return {
            category: embedding.tolist()
            for category, embedding in zip(categories, embeddings)
        }

    def embed_reviews(
            self,
            reviews: list[tuple[str, str, str]]
    ) -> tuple[list[list[float]], ...]:
        aggregated = [
            f'TITLE:{title} - CONTENT:{content}'
            for _, title, content in reviews
        ]

        Embedder._logger.info("Embedding review titles...")
        title_embeddings = self.review_embedder.encode(
            [title for _, title, _ in reviews],
            show_progress_bar = True
        )

        Embedder._logger.info("Embedding review content...")
        content_embeddings = self.review_embedder.encode(
            [content for _, _, content in reviews],
            show_progress_bar = True
        )

        Embedder._logger.info("Embedding full reviews...")
        review_embeddings = self.review_embedder.encode(
            aggregated,
            show_progress_bar = True
        )

        return tuple([
            [vec.tolist() for vec in embeddings]
            for embeddings
            in (title_embeddings, content_embeddings, review_embeddings)
        ])

    def embed_podcasts(self) -> None: ...
