# Install vector similarity search extension
import duckdb
duckdb.install_extension("vss")

# testing start
from .podcastsearch import PodcastSearch
PodcastSearch("../data/light.db")
