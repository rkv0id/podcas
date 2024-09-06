from logging import getLogger
from typing import Optional
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import torch

from podcas import lib_device
from podcas.ml import Summarizer
from ._textdataset import TextDataset


class Embedder:
    """
    A class for embedding text data using HuggingFace pre-trained language models.

    Attributes:
        cat_tokenizer: Tokenizer for the category model.
        cat_model: Pre-trained model for category embeddings.
        rev_tokenizer: Tokenizer for the review model.
        rev_model: Pre-trained model for review embeddings.
        pod_tokenizer: Tokenizer for the podcast model.
        pod_model: Pre-trained model for podcast embeddings.
        summarizer: Optional summarizer for text preprocessing.
        max_length: Maximum sequence length for tokenization.
        batch_size: Batch size for data loading.
        model_names: Dictionary mapping model types to their names.
    """

    _logger = getLogger(f"{__name__}.{__qualname__}")

    def __init__(
            self,
            category_model: str,
            review_model: str,
            podcast_model: str,
            summarizer: Optional[Summarizer] = None,
            max_length: int = 256,
            batch_size: int = 32
    ):
        """
        Initializes the Embedder with pre-trained models and tokenizers.

        Args:
            category_model: Path or name of the pre-trained category model.
            review_model: Path or name of the pre-trained review model.
            podcast_model: Path or name of the pre-trained podcast model.
            summarizer: Optional Summarizer object for text summarization.
            max_length: Maximum token length for input sequences.
            batch_size: Batch size for processing inputs.
        """

        self.cat_tokenizer = AutoTokenizer.from_pretrained(category_model)
        self.cat_model = AutoModel.from_pretrained(category_model).to(lib_device)
        self.cat_model.eval()

        self.rev_tokenizer = AutoTokenizer.from_pretrained(review_model)
        self.rev_model = AutoModel.from_pretrained(review_model).to(lib_device)
        self.rev_model.eval()

        self.pod_tokenizer = AutoTokenizer.from_pretrained(podcast_model)
        self.pod_model = AutoModel.from_pretrained(podcast_model).to(lib_device)
        self.pod_model.eval()

        self.summarizer = summarizer

        self.max_length = max_length
        self.batch_size = batch_size
        self.model_names = {
            'category': category_model,
            'review': review_model,
            'podcast': podcast_model
        }

    def embed_text(self, texts: list[str], tokenizer, model) -> list[torch.Tensor]:
        """
        Embeds a list of texts using the specified tokenizer and model.

        Args:
            texts: List of texts to embed.
            tokenizer: Tokenizer corresponding to the model.
            model: Pre-trained model used for embedding.

        Returns:
            A list of embeddings as torch Tensors.
        """

        # summarize while needed
        if self.summarizer:
            summarized = texts.copy()
            try: self.summarizer.summarize_inplace(summarized)
            except Exception as e:
                Embedder._logger.error(
                    f"Error occured while summarizing: {e}",
                    exc_info=True
                )
        else: summarized = texts

        # batch-process for gpu-efficiency
        dataset = TextDataset(summarized)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        all_embeddings = []
        for batch in tqdm(
                dataloader,
                total=len(dataloader),
                desc="_embedding"
        ):
            encoded_input = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length
            ).to(lib_device)

            with torch.no_grad(): model_output = model(**encoded_input)

            token_embeddings = model_output[0]
            attention_mask = encoded_input['attention_mask']
            input_mask_expanded = (
                attention_mask
                    .unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float()
            )

            # mean pool (weigh by attention mask)
            batch_embeddings = (
                torch.sum(token_embeddings * input_mask_expanded, 1)
                / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            )

            all_embeddings.extend(batch_embeddings.cpu().unbind(0))

        return all_embeddings

    def embed_categories(self, categories: list[str]) -> dict[str, list[float]]:
        """
        Embeds a list of category texts.

        Args:
            categories: List of category texts.

        Returns:
            A dictionary mapping category name to its corresponding embedding vector.
        """
        Embedder._logger.info("Embedding categories...")
        embeddings = self.embed_text(
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
            reviews: list[tuple[int, str, str]]
    ) -> list[list[float]]:
        """
        Embeds a list of reviews.

        Args:
            reviews: List of reviews, each represented as a tuple (id, title, content).

        Returns:
            A list of embedding vectors for the reviews.
        """
        aggregated = [
            f'TITLE:{title} - CONTENT:{content}'
            for _, title, content in reviews
        ]

        Embedder._logger.info("Embedding reviews...")
        embeddings = self.embed_text(
            aggregated,
            self.rev_tokenizer,
            self.rev_model
        )

        return [vec.tolist() for vec in embeddings]

    def embed_episodes(
            self,
            descriptions: list[str],
            nested_rev_vectors: list[list[list[float]]]
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Embeds episode descriptions and their associated review vectors.

        Args:
            descriptions: List of episode descriptions.
            nested_rev_vectors: Nested list of review embeddings for each episode.

        Returns:
            A tuple containing the list of description embeddings and averaged review embeddings.
        """
        Embedder._logger.info("Embedding episodes...")
        desc_embeddings = self.embed_text(
            descriptions,
            self.rev_tokenizer,
            self.rev_model
        )

        rev_embeddings = []
        for review_vectors in nested_rev_vectors:
            rev_embeddings.append(
                torch.mean(
                    torch.tensor(review_vectors, dtype=torch.float32),
                    dim=0
                ).tolist()
            )

        return (
            [vec.tolist() for vec in desc_embeddings],
            rev_embeddings
        )

    def embed_podcasts(
            self,
            nested_desc_vectors: list[list[list[float]]],
            nested_cat_vectors: list[list[list[float]]],
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        Embeds podcast descriptions and categories.

        Args:
            nested_desc_vectors: Nested list of description embeddings.
            nested_cat_vectors: Nested list of category embeddings.

        Returns:
            A tuple containing the averaged description embeddings and category embeddings.
        """
        desc_embeddings = []
        for desc_vectors in nested_desc_vectors:
            desc_embeddings.append(
                torch.mean(
                    torch.tensor(desc_vectors, dtype=torch.float32),
                    dim=0
                ).tolist()
            )

        cat_embeddings = []
        for cat_vectors in nested_cat_vectors:
            cat_embeddings.append(
                torch.mean(
                    torch.tensor(cat_vectors, dtype=torch.float32),
                    dim=0
                ).tolist()
            )

        return desc_embeddings, cat_embeddings
