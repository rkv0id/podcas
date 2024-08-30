# Install vector similarity search extension
import duckdb
duckdb.install_extension("vss")

# testing start
from .podcastsearch import PodcastSearch
from .reviewsearch import ReviewSearch

__all__ = ['PodcastSearch', 'ReviewSearch']

# default_embedder = 'multi-qa-MiniLM-L6-cos-v1'
# print(
#     PodcastSearch()
#         .load(source="../data/mid.db")
#         .using(
#             category_model = default_embedder,
#             podcast_model = default_embedder,
#             review_model = default_embedder)
#         .top(4)
#         .by_rating(3, max=4)
#         .by_description('tales of the free cities')
#         .get()
# )
