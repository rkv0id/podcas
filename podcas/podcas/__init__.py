# Install vector similarity search extension
import duckdb
duckdb.install_extension("vss")

# testing start
default_embedder = 'multi-qa-MiniLM-L6-cos-v1'
from .podcastsearch import PodcastSearch
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
