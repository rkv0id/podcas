# Podcas: Podcast Search and Recommendation Engine

`podcas` is a powerful podcast search and recommendation library designed to help users discover podcasts and episodes based on titles, descriptions, categories, ratings, and more. The library leverages modern NLP techniques, including embeddings and sentiment analysis, to provide personalized search results.

## Key Features

- **Podcast Search:** Search for podcasts by title, author, category, and description.
- **Episode Search:** Find specific episodes based on reviews, descriptions, and ratings.
- **Review Search:** Find specific episodes reviews based on their content and sentiment.
- **Advanced Embeddings:** Supports HuggingFace-available embedding models for precise text similarity.
- **Sentiment Analysis:** Analyze reviews sentiments to boost search filtering by sentiment.
- **Fuzzy Matching:** Support for fuzzy matching on titles and authors.

## Installation

To use `podcas`, you need Python 3.8 or higher and the Poetry package manager. If Poetry is not installed, you can get it from [Poetry's official website](https://python-poetry.org/docs/#installation).

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/podcas.git
cd podcas
```

### Step 2: Install Dependencies
Ensure you have Poetry installed, then run:
```bash
poetry install
```
This will install `podcas` along with all required dependencies specified in the pyproject.toml file.

## Usage
`PodcastSearch`, `EpisodeSearch`, and `ReviewSearch` allow you to find podcasts, episodes of podcasts, and even reviews based on various criteria such as title, author, and ratings. It also supports similarity-based search for some semantic criteria like category, and description.
```python
from podcas import PodcastSearch, EpisodeSearch

# Configure (here, some of) the models used for embedding and sentiment analysis
search = PodcastSearch().using(
    category_model="sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking",
    podcast_model="sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking"
)

# Initialize search with a data source
search = search.load(source="../data/mid.db")

# Filter by top podcasts, ratings, and categories
podcasts = (
    search
      .top(5)
      .by_rating(min=4.0)
      .by_category("comedy")
      .get()
)

# Display results
for title, author, rating, similarity_score in podcasts:
    print(f"Title: {title}, Author: {author}, Rating: {rating}, Score: {similarity_score}")

# Example episode search by description
episode_results = (
    EpisodeSearch()
        .load(source="../data/mid.db")
        .by_description("inspiring stories of success")
        .top(3)
        .get()
)

for title, author, itunes_id, rating, similarity_score in episode_results:
    print(f"Podcast: {podcast_title}, Author: {author}, iTunes ID: {itunes_id}, Rating: {rating}, Score: {similarity_score}")
```

## Configuration and Models
`podcas` uses pre-trained models for embeddings, summarization, and sentiment analysis. The default models are:
- Embedding Model: `sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking`
- Sentiment Model: `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`
- Summarization Model: `google/pegasus-xsum`

You can specify different models through the `.using()` method according to your needs.

## Device Support
`podcas` automatically selects the best available device:
- CUDA: For NVIDIA GPUs.
- MPS: For Apple M/x GPUs.
- CPU: When no GPU is available.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue on GitHub to suggest improvements, report bugs, or propose new features.

## Contact
For questions or support, please contact ramy.kader.rk@gmail.com or open an issue on GitHub.
