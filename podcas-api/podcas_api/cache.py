from asyncio import Lock, to_thread
from collections import OrderedDict
from podcas.data import DataStore
from podcas.ml import Embedder, Mooder


# Currently only handles (file -> DataStore) relationship
# although DataStore atomicity is defined by (file, **models)
# TODO: Transform key to (file, **models) and handle file duplication
# TODO: consider creating a data pull/clone and file server-side caching
# to avoid collisions in data sources (due to library mutating data)
class DataStoreLRUCache:
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
