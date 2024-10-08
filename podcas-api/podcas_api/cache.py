from asyncio import Lock, to_thread
from collections import OrderedDict
from typing import Optional
from podcas.data import DataStore
from podcas.ml import Embedder, Mooder, Summarizer


# Currently only handles (file -> DataStore) relationship
# although DataStore atomicity is defined by (file, **models)
# TODO: Transform key to (file, **models) and handle file duplication
# TODO: consider creating a data pull/clone and file server-side caching
# to avoid collisions in data sources (due to library mutating data)
class DataStoreLRUCache:
    "Cache implementation for DataStore objects with LRU eviction policy."

    _instance = None
    _instance_lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            async with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, capacity: int) -> None:
        self.__lock = Lock()
        self.__capacity = capacity
        self.__size = 0
        self.__cache: OrderedDict[str, DataStore] = OrderedDict()
        self.__cache_lock: dict[str, Lock] = {}

    async def get(
            self,
            path: str,
            embedder: Embedder,
            mooder: Mooder
    ) -> DataStore:
        async with self.__lock:
            if path in self.__cache:
                self.__cache.move_to_end(path)
                return self.__cache[path]

            if path not in self.__cache_lock:
                self.__cache_lock[path] = Lock()

        async with self.__cache_lock[path]:
            async with self.__lock:
                if path in self.__cache:
                    self.__cache.move_to_end(path)
                    return self.__cache[path]

            datastore = await to_thread(DataStore, path, embedder, mooder)

            async with self.__lock:
                self.__cache[path] = datastore
                self.__cache.move_to_end(path)
                self.__size += 1
                if self.__size > self.__capacity:
                    oldest, _ = self.__cache.popitem(last=False)
                    del self.__cache_lock[oldest]

            return datastore

class EmbedderLRUCache:
    "Cache implementation for Embedder objects with LRU eviction policy."

    _instance = None
    _instance_lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            async with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, capacity: int) -> None:
        self.__lock = Lock()
        self.__capacity = capacity
        self.__size = 0
        self.__cache: OrderedDict[
            tuple[str, str, str, Optional[str]],
            Embedder
        ] = OrderedDict()
        self.__cache_lock: dict[
            tuple[str, str, str, Optional[str]],
            Lock
        ] = {}

    async def get(
            self,
            category_model: str,
            review_model: str,
            podcast_model: str,
            summarizer: Optional[str]
    ) -> Embedder:
        key = (
            category_model,
            review_model,
            podcast_model,
            summarizer
        )

        async with self.__lock:
            if key in self.__cache:
                self.__cache.move_to_end(key)
                return self.__cache[key]

            if key not in self.__cache_lock:
                self.__cache_lock[key] = Lock()

        async with self.__cache_lock[key]:
            async with self.__lock:
                if key in self.__cache:
                    self.__cache.move_to_end(key)
                    return self.__cache[key]

            summarizer_instance = (
                None if not summarizer
                else await to_thread(Summarizer, summarizer)
            )
            embedder = await to_thread(
                Embedder,
                category_model,
                review_model,
                podcast_model,
                summarizer_instance
            )

            async with self.__lock:
                self.__cache[key] = embedder
                self.__cache.move_to_end(key)
                self.__size += 1
                if self.__size > self.__capacity:
                    oldest, _ = self.__cache.popitem(last=False)
                    del self.__cache_lock[oldest]

            return embedder

class MooderLRUCache:
    "Cache implementation for Mooder objects with LRU eviction policy."

    _instance = None
    _instance_lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            async with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, capacity: int) -> None:
        self.__lock = Lock()
        self.__capacity = capacity
        self.__size = 0
        self.__cache: OrderedDict[
            tuple[str, Optional[str]],
            Mooder
        ] = OrderedDict()
        self.__cache_lock: dict[
            tuple[str, Optional[str]],
            Lock
        ] = {}

    async def get(
            self,
            model: str,
            summarizer: Optional[str]
    ) -> Mooder:
        key = (model, summarizer)

        async with self.__lock:
            if key in self.__cache:
                self.__cache.move_to_end(key)
                return self.__cache[key]

            if key not in self.__cache_lock:
                self.__cache_lock[key] = Lock()

        async with self.__cache_lock[key]:
            async with self.__lock:
                if key in self.__cache:
                    self.__cache.move_to_end(key)
                    return self.__cache[key]

            summarizer_instance = (
                None if not summarizer
                else await to_thread(Summarizer, summarizer)
            )
            mooder = await to_thread(Mooder, model, summarizer_instance)

            async with self.__lock:
                self.__cache[key] = mooder
                self.__cache.move_to_end(key)
                self.__size += 1
                if self.__size > self.__capacity:
                    oldest, _ = self.__cache.popitem(last=False)
                    del self.__cache_lock[oldest]

            return mooder
