from logging import getLogger
logger = getLogger(__name__)

# Install vector similarity search extension
import duckdb
duckdb.install_extension("vss")

from torch import device, cuda
from torch.backends import mps

if cuda.is_available():
    lib_device = device("cuda")
    logger.info("Using NVIDIA GPU [CUDA]")
elif mps.is_available():
    lib_device = device("mps")
    logger.info("Using Apple M/x GPU [MPS]")
else:
    lib_device = device("cpu")
    logger.info("Using CPU")

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking"
DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
DEFAULT_SUMMARIZE_MODEL = "google/pegasus-xsum"

from .search import PodcastSearch, ReviewSearch, EpisodeSearch
from .data import Review, Episode, Podcast
__all__ = [
    'PodcastSearch',
    'ReviewSearch',
    'EpisodeSearch',
    'Review',
    'Episode',
    'Podcast'
]


# EXAMPLE USAGE
# from podcas import PodcastSearch
# default_model = 'multi-qa-MiniLM-L6-cos-v1'
# print(
#     PodcastSearch()
#         .load(source="../data/mid.db")
#         .using(
#             category_model = default_model,
#             podcast_model = default_model,
#             review_model = default_model,
#             sentiment_model = default_model)
#         .top(4)
#         .by_rating(3, max=4)
#         .by_category('fiction')
#         .get()
# )
