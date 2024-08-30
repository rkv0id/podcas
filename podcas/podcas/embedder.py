from logging import getLogger
from sentence_transformers import SentenceTransformer


class Embedder:
    DEFAULT_VEC_SIZE = 128
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
        self.model_names = {
            'category': category_model,
            'review': review_model,
            'podcast': podcast_model
        }

    def embed_categories(self, categories: list[str]) -> dict[str, list[float]]:
        Embedder._logger.info("Embedding categories...")
        embeddings = self.cat_embedder.encode(
            categories,
            show_progress_bar=True,
            batch_size=128
        )

        return {
            category: embedding.tolist()
            for category, embedding in zip(categories, embeddings)
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
            show_progress_bar = True,
            batch_size=32
        )

        return [vec.tolist() for vec in embeddings]

    def embed_podcasts(
            self,
            descriptions: list[list[str]],
            reviews: list[list[tuple[str, str]]],
            categories: list[list[str]],
    ) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
        agg_descriptions = [
            '\n'.join([
                f"EP: {description}"
                for description in descs
            ])
            for descs in descriptions
        ]

        agg_reviews = [
            '\n'.join([
                f"REV: title:{title} - content:{content}"
                for title, content in review_pairs
            ])
            for review_pairs in reviews
        ]

        agg_categories = [
            ', '.join([category for category in podcast_categories])
            for podcast_categories in categories
        ]

        Embedder._logger.info("Embedding podcasts descriptions...")
        desc_embeds = self.pod_embedder.encode(
            agg_descriptions,
            show_progress_bar=True,
            batch_size=32
        ) if len(agg_descriptions) > 0 else []

        Embedder._logger.info("Embedding podcasts reviews...")
        rev_embeds = self.rev_embedder.encode(
            agg_reviews,
            show_progress_bar=True,
            batch_size=32
        ) if len(agg_reviews) > 0 else []

        Embedder._logger.info("Embedding podcasts categories...")
        cat_embeds = self.cat_embedder.encode(
            agg_categories,
            show_progress_bar=True,
            batch_size=128
        ) if len(agg_categories) > 0 else []

        return (
            [vec.tolist() for vec in desc_embeds],
            [vec.tolist() for vec in rev_embeds],
            [vec.tolist() for vec in cat_embeds]
        )
