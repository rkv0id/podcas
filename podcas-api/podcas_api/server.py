from fastapi import FastAPI
from podcas import ReviewSearch, EpisodeSearch, PodcastSearch

from .cache import DataStoreLRUCache, EmbedderLRUCache, MooderLRUCache
from .queries import (
    ReviewSearchParams,
    EpisodeSearchParams,
    PodcastSearchParams
)


MAX_CACHE_SIZE = 20
datastore_cache = DataStoreLRUCache(capacity=MAX_CACHE_SIZE)
embedder_cache = EmbedderLRUCache(capacity=MAX_CACHE_SIZE)
mooder_cache = MooderLRUCache(capacity=MAX_CACHE_SIZE)
app = FastAPI()

@app.post("/search/reviews")
async def search_reviews(params: ReviewSearchParams):
    embedder = await embedder_cache.get(
        params.category_model,
        params.review_model,
        params.podcast_model,
        params.summarizer_model
    )

    mooder = await mooder_cache.get(
        params.sentiment_model,
        params.summarizer_model
    )

    db = await datastore_cache.get(params.db_file, embedder, mooder)

    search = (
        ReviewSearch(db, embedder, mooder)
            .top(params.top)
            .rating_boosted(params.boost_by_rating)
            .by_rating(min=params.min_rating, max=params.max_rating)
    )

    match params.sentiment:
        case 'negative': search = search.negative()
        case 'neutral': search = search.neutral()
        case 'positive': search = search.positive()
        case _: pass

    if params.text: search = search.by_query(params.text)

    return search.get()


@app.post("/search/episodes")
async def search_episodes(params: EpisodeSearchParams):
    embedder = await embedder_cache.get(
        params.category_model,
        params.review_model,
        params.podcast_model,
        params.summarizer_model
    )

    mooder = await mooder_cache.get(
        params.sentiment_model,
        params.summarizer_model
    )

    db = await datastore_cache.get(params.db_file, embedder, mooder)

    search = (
        EpisodeSearch(db, embedder, mooder)
            .top(params.top)
            .rating_boosted(params.boost_by_rating)
            .by_rating(min=params.min_rating, max=params.max_rating)
    )

    if params.title:
        search = search.by_title(params.title, fuzzy=params.fuzzy_title)

    if params.author:
        search = search.by_author(params.author, fuzzy=params.fuzzy_author)

    if params.review_text:
        search = search.by_review(params.review_text)

    if params.description_text:
        search = search.by_description(params.description_text)

    return search.get()


@app.post("/search/podcasts")
async def search_podcasts(params: PodcastSearchParams):
    embedder = await embedder_cache.get(
        params.category_model,
        params.review_model,
        params.podcast_model,
        params.summarizer_model
    )

    mooder = await mooder_cache.get(
        params.sentiment_model,
        params.summarizer_model
    )

    db = await datastore_cache.get(params.db_file, embedder, mooder)

    search = (
        PodcastSearch(db, embedder, mooder)
            .top(params.top)
            .rating_boosted(params.boost_by_rating)
            .by_rating(min=params.min_rating, max=params.max_rating)
    )

    if params.title:
        search = search.by_title(params.title, fuzzy=params.fuzzy_title)

    if params.author:
        search = search.by_author(params.author, fuzzy=params.fuzzy_author)

    if params.category:
        search = search.by_category(params.category)

    if params.description_text:
        search = search.by_description(params.description_text)

    return search.get()
