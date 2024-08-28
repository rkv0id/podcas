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
            show_progress_bar=True
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
            show_progress_bar = True
        )

        return [vec.tolist() for vec in embeddings]

    def embed_podcasts(
            self,
            descriptions: list[list[str]],
            reviews: list[list[tuple[str, str]]]
    ) -> tuple[list[list[float]], list[list[float]]]:
        agg_descriptions = [
            '\n'.join([
                f"EP: {description}"
                for description in descs
            ])
            for descs in descriptions
        ]

        agg_reviews = [
            '\n'.join([
                f"title:{r_title} - content:{r_content}"
                for r_title, r_content in review_pairs
            ])
            for review_pairs in reviews
        ]

        Embedder._logger.info("Embedding podcasts descriptions...")
        desc_embeds = self.pod_embedder.encode(
            agg_descriptions,
            show_progress_bar=True
        )

        Embedder._logger.info("Embedding podcasts reviews...")
        rev_embeds = self.pod_embedder.encode(
            agg_reviews,
            show_progress_bar=True
        )

        return (
            [vec.tolist() for vec in desc_embeds],
            [vec.tolist() for vec in rev_embeds]
        )
