from dataclasses import dataclass


@dataclass
class Review:
    title: str
    content: str
    rating: float
    similarity_score: float

@dataclass
class Episode:
    title: str
    author: str
    itunes_id: str
    rating: float
    similarity_score: float

@dataclass
class Podcast:
    title: str
    author: str
    rating: float
    similarity_score: float
