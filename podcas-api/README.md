# Podcas-API

**Podcas-API** is a FastAPI-based web application that provides an interface to explore podcast databases using the `podcas` library. It enables users to search for reviews, episodes, and podcasts based on various criteria using machine learning models to analyze content sentiment, categories, and summaries.

## Features

- **Search Reviews**: Find podcast reviews based on rating, sentiment, and text queries.
- **Search Episodes**: Locate episodes using filters such as title, author, review text, and description.
- **Search Podcasts**: Explore podcasts by filtering based on title, author, category, and description.
- **Caching System**: Efficient caching using LRU (Least Recently Used) cache implementation for `DataStore`, `Embedder`, and `Mooder` objects to optimize performance and reduce redundant computations.

## Installation

### Prerequisites

- Python 3.8+
- Poetry for dependency management (optional but recommended)
- FastAPI
- Uvicorn (for running the FastAPI server)

### Step-by-step Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:rkv0id/periph-podcas.git
   cd periph-podcas/podcas-ai
   ```

2. **Install dependencies using Poetry**
    ```bash
    poetry install
    ```

3. **Run the server**
   ```bash
   fastapi run podcas_api/server
   ```
   The API will be accessible at http://localhost:8000.

## Documentation
Make sure to checkout http://localhost:8000/docs or http://localhost:8000/redoc for the interactive API docs where you can even try out the different endpoints it exposes.

The API uses an *LRU* (Least Recently Used) caching strategy to efficiently manage resources:

- **DataStore Cache**: Manages the lifecycle of data access stores based on file paths (and other sources *- TBD*) and associated ML models.
- **Embedder Cache**: Handles caching of embedding models based on HuggingFace models configurations.
- **Mooder Cache**: Caches sentiment analyzer instances based on the selected sentiment analysis model.

The caching system uses asynchronous locks to prevent race conditions and ensure safe access across concurrent requests.

Please check out the Podcas library for more details about the inner workings of this service as this is just an API layer exposing that library's functionality under different endpoints.

## TODO
- [ ] Cache key from `file` to `(file, **models)`.
- [ ] File server-side caching to avoid mutation collisions.
- [ ] Validate data sources based on required schema
- [ ] Logging :mag:
- [ ] Testing :sleepy:
- [ ] Stress test & benchmark endpoints

## Contributing
Contributions are welcome! Please submit a pull request or open an issue on GitHub to suggest improvements, report bugs, or propose new features.

## Contact
For questions or support, please contact ramy.kader.rk@gmail.com or open an issue on GitHub.
