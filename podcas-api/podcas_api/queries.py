from os.path import isfile
from huggingface_hub import model_info
from typing import Literal, Optional, Self
from pydantic import BaseModel, Field, model_validator


class BaseSearchParams(BaseModel):
    db_file: str = Field(description="Data source file to embed/explore")
    category_model: Optional[str] = Field(None, description="Categories embedding model")
    review_model: Optional[str] = Field(None, description="Reviews embedding model")
    podcast_model: Optional[str] = Field(None, description="Podcasts embedding model")
    summarizer_model: Optional[str] = Field(None, description="Summarization model")
    sentiment_model: Optional[str] = Field(None, description="Sentiment analysis model")
    top: int = Field(
        default=3,
        ge=1, le=100,
        description="Number of top matching results to retrieve")
    min_rating: float = Field(
        default=0.0,
        ge=0.0, le=5.0,
        description="Minimum rating filter")
    max_rating: float = Field(
        default=5.0,
        ge=0.0, le=5.0,
        description="Maximum rating filter")
    boost_by_rating: bool = Field(
        default=False,
        description="Bias result towards higher-rated reviews"
    )

    @model_validator(mode="after")
    def validate(self) -> Self:
        if self.min_rating > self.max_rating:
            raise ValueError("'min_rating' must be lower than or equal to 'max_rating'!")

        if not self.db_file or not isfile(self.db_file):
            raise ValueError("Unable to locate data source or is currently unavailable!")
        
        if self.category_model: BaseSearchParams.validate_model(self.category_model)
        if self.review_model: BaseSearchParams.validate_model(self.review_model)
        if self.podcast_model: BaseSearchParams.validate_model(self.podcast_model)
        if self.summarizer_model: BaseSearchParams.validate_model(self.summarizer_model)
        if self.sentiment_model: BaseSearchParams.validate_model(self.sentiment_model)

        return self

    @staticmethod
    def validate_model(model_name: str) -> None:
        try: _ = model_info(model_name)
        except:
            raise ValueError(f"The model {model_name} doesn't exist neither locally nor on HuggingFace public hub.")


class ReviewSearchParams(BaseSearchParams):
    sentiment: Optional[Literal["neutral", "positive", "negative"]] = Field(
        None, description="Review sentiment filter - options available according to model used"
    )
    text: Optional[str] = Field(
        None, max_length=150,
        description="Text to search for similar reviews")

    @model_validator(mode="before")
    @classmethod
    def transform_sentiment(cls, values):
        sentiment = values.get('sentiment', '')
        if sentiment: values['sentiment'] = sentiment.lower()
        return values


class EpisodeSearchParams(BaseSearchParams):
    title: Optional[str] = Field(
        None, max_length=75,
        description="Title filter for the episodes' podcast")
    author: Optional[str] = Field(
        None, max_length=75,
        description="Author filter for the episodes' podcast")
    fuzzy_title: bool = Field(False, description="Fuzzy search on the title field")
    fuzzy_author: bool = Field(False, description="Fuzzy search on the author field")
    review_text: Optional[str] = Field(
        None, max_length=150,
        description="Smart review-based filter - by similarity")
    description_text: Optional[str] = Field(
        None, max_length=150,
        description="Smart description-based filter - by similarity")

    @model_validator(mode="before")
    @classmethod
    def transform_episode(cls, values):
        title = values.get('title', '')
        author = values.get('author', '')
        if title: values['title'] = title.lower()
        if author: values['author'] = author.lower()
        return values


class PodcastSearchParams(BaseSearchParams):
    title: Optional[str] = Field(
        None, max_length=75,
        description="Podcast title filter")
    author: Optional[str] = Field(
        None, max_length=75,
        description="Podcast author filter")
    fuzzy_title: bool = Field(False, description="Fuzzy search on the title field")
    fuzzy_author: bool = Field(False, description="Fuzzy search on the author field")
    category: Optional[str] = Field(
        None, max_length=75,
        description="Smart category-based filter - by similarity")
    description_text: Optional[str] = Field(
        None, max_length=150,
        description="Smart description-based filter - by similarity")

    @model_validator(mode="before")
    @classmethod
    def transform_podcast(cls, values):
        title = values.get('title', '')
        author = values.get('author', '')
        category = values.get('category', '')
        if title: values['title'] = title.lower()
        if author: values['author'] = author.lower()
        if category: values['category'] = category.lower()
        return values
