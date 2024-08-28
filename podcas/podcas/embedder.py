from logging import getLogger
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


class Embedder:
    _logger = getLogger(f"{__name__}.{__qualname__}")

    def __init__(
            self,
            category_model: str,
            review_model: str,
            podcast_model: str
    ):
        self.cat_embedder = SentenceTransformer(category_model)
        self.rev_embedder = SentenceTransformer(review_model)
        self.pod_embedder = SentenceTransformer(podcast_model)

        self.cat_reducer = PCA(n_components='mle')
        self.rev_reducer = PCA(n_components='mle')
        self.pod_about_reducer = PCA(n_components='mle')
        self.pod_review_reducer = PCA(n_components='mle')

        self.model_names = {
            'category': category_model,
            'review': review_model,
            'podcast': podcast_model
        }

    def embed_categories(self, categories: list[str]) -> dict[str, list[float]]:
        Embedder._logger.info("Embedding categories...")
        embeddings = self.cat_embedder.encode(
            categories,
            show_progress_bar=True
        )

        Embedder._logger.info("Reducing categories embeddings...")
        reduced_embeddings = self.cat_reducer.fit_transform(embeddings)

        return {
            category: embedding.tolist()
            for category, embedding in zip(categories, reduced_embeddings)
        }

    def embed_reviews(
            self,
            reviews: list[tuple[str, str, str]]
    ) -> list[list[float]]:
        aggregated = [
            f'TITLE:{title} - CONTENT:{content}'
            for _, title, content in reviews
        ]

        Embedder._logger.info("Embedding reviews...")
        embeddings = self.rev_embedder.encode(
            aggregated,
            show_progress_bar = True
        )

        Embedder._logger.info("Reducing reviews embeddings...")
        reduced_embeddings = self.rev_reducer.fit_transform(embeddings)

        return [vec.tolist() for vec in reduced_embeddings]

    def embed_podcasts(self) -> None: ...
