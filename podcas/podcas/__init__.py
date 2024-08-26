import duckdb

from .podcastsearch import PodcastSearch

# Vector similarity search extension
duckdb.execute("install vss")
duckdb.execute("load vss")

PodcastSearch("../data/database.db")
