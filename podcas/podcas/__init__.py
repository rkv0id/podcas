import duckdb

from .podcastsearch import PodcastSearch

# Vector similarity search extension
duckdb.install_extension("vss")

PodcastSearch("../data/light.db")
