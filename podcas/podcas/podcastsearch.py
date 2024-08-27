from threading import Lock

from .datastore import Datastore

class PodcastSearch:
    __instance = None
    __lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls.__lock:
            if not cls.__instance:
                cls.__instance = super(PodcastSearch, cls).__new__(cls)
        return cls.__instance

    def __init__(self, path: str):
        self.source = path
        self.__db = Datastore(path)
