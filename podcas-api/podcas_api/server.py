from json import dumps
from fastapi import FastAPI
from dataclasses import asdict
from podcas import ReviewSearch, EpisodeSearch, PodcastSearch

from .cache import DataStoreLRUCache
from .queries import (
    ReviewSearchParams,
    EpisodeSearchParams,
    PodcastSearchParams
)


MAX_CACHE_SIZE = 20
cache = DataStoreLRUCache(capacity=20)
app = FastAPI()

@app.post("/search/reviews")
async def search_reviews(params: ReviewSearchParams):
    search = ReviewSearch().using(
        category_model=params.category_model,
        review_model=params.review_model,
        podcast_model=params.podcast_model,
        mooder_model=params.sentiment_model,
        summary_model=params.summarizer_model
    )

    search._db = await cache.get(
        params.db_file,
        search._embedder,
        search._mooder
    )

    search = (
        search
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
    search = EpisodeSearch().using(
        category_model=params.category_model,
        review_model=params.review_model,
        podcast_model=params.podcast_model,
        mooder_model=params.sentiment_model,
        summary_model=params.summarizer_model
    )

    search._db = await cache.get(
        params.db_file,
        search._embedder,
        search._mooder
    )

    search = (
        search
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
    search = PodcastSearch().using(
        category_model=params.category_model,
        review_model=params.review_model,
        podcast_model=params.podcast_model,
        mooder_model=params.sentiment_model,
        summary_model=params.summarizer_model
    )

    search._db = await cache.get(
        params.db_file,
        search._embedder,
        search._mooder
    )

    search = (
        search
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
