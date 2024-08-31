from logging import getLogger
from transformers import AutoModel, AutoTokenizer
import torch

from . import lib_device


class Embedder:
    DEFAULT_VEC_SIZE = 128
    _logger = getLogger(f"{__name__}.{__qualname__}")

    def __init__(
            self,
            category_model: str,
            review_model: str,
            podcast_model: str
    ):
        self.cat_tokenizer = AutoTokenizer.from_pretrained(category_model)
        self.cat_model = AutoModel.from_pretrained(category_model).to(lib_device)

        self.rev_tokenizer = AutoTokenizer.from_pretrained(review_model)
        self.rev_model = AutoModel.from_pretrained(review_model).to(lib_device)

        self.pod_tokenizer = AutoTokenizer.from_pretrained(podcast_model)
        self.pod_model = AutoModel.from_pretrained(podcast_model).to(lib_device)

        self.model_names = {
            'category': category_model,
            'review': review_model,
            'podcast': podcast_model
        }

    @staticmethod
    def embed_text(texts: list[str], tokenizer, model) -> torch.Tensor:
        Embedder._logger.info("_tokenizing input...")
        encoded_input = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=64
        ).to(lib_device)

        Embedder._logger.info("_embedding tokenized input...")
        with torch.no_grad(): model_output = model(**encoded_input)

        token_embeddings = model_output[0]
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = (
            attention_mask
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
        )

        embeddings = (
            torch.sum(token_embeddings * input_mask_expanded, 1)
            / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        )

        return embeddings.cpu()

    def embed_categories(self, categories: list[str]) -> dict[str, list[float]]:
        Embedder._logger.info("Embedding categories...")
        embeddings = Embedder.embed_text(
            categories,
            self.cat_tokenizer,
            self.cat_model
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
        embeddings = Embedder.embed_text(
            aggregated,
            self.rev_tokenizer,
            self.rev_model
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
                f"EPISODE: {description}"
                for description in descs
            ])
            for descs in descriptions
        ]

        agg_reviews = [
            '\n'.join([
                f"REVIEW: TITLE:{title} - CONTENT:{content}"
                for title, content in review_pairs
            ])
            for review_pairs in reviews
        ]

        agg_categories = [
            ', '.join([category for category in podcast_categories])
            for podcast_categories in categories
        ]

        Embedder._logger.info("Embedding podcasts descriptions...")
        desc_embeds = Embedder.embed_text(
            agg_descriptions,
            self.pod_tokenizer,
            self.pod_model
        ) if len(agg_descriptions) > 0 else []

        Embedder._logger.info("Embedding podcasts reviews...")
        rev_embeds = Embedder.embed_text(
            agg_reviews,
            self.rev_tokenizer,
            self.rev_model
        ) if len(agg_reviews) > 0 else []

        Embedder._logger.info("Embedding podcasts categories...")
        cat_embeds = Embedder.embed_text(
            agg_categories,
            self.cat_tokenizer,
            self.cat_model
        ) if len(agg_categories) > 0 else []

        return (
            [vec.tolist() for vec in desc_embeds],
            [vec.tolist() for vec in rev_embeds],
            [vec.tolist() for vec in cat_embeds]
        )
