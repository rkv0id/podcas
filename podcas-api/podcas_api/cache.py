from typing import Optional, Self
from podcas.data import DataStore


class Node:
    def __init__(
        self,
        key: str,
        val: Optional[DataStore] = None,
        prv: Optional[Self] = None,
        nxt: Optional[Self] = None
    ):
        self.key = key
        self.val = val
        self.prv = prv
        self.nxt = nxt

class LRUCache:
    def __init__(self, capacity: int):
        assert capacity > 0
        self.__capacity: int = capacity
        self.__size: int = 0
        self.__cache: dict[str, Node] = {}
        self.__oldest: Node = Node("oldest__")
        self.__recent: Node = Node("__recent")
        self.__oldest.nxt = self.__recent
        self.__recent.prv = self.__oldest

    def get(self, key: str) -> Optional[DataStore]:
        if key in self.__cache:
            node = self.__cache[key]
            self.__remove(node)
            self.__insert(node)
            return node.val
        else: return None

    def put(self, key: str, value: DataStore) -> None:
        if key not in self.__cache: self.__size += 1
        else: self.__remove(self.__cache[key])
        self.__cache[key] = Node(key, value)
        self.__insert(self.__cache[key])

        if self.__size > self.__capacity:
            lru = self.__oldest.nxt
            if lru:
                self.__remove(lru)
                del self.__cache[lru.key], lru
                self.__size -= 1

    def __remove(self, node: Node) -> None:
        if node.prv: node.prv.nxt = node.nxt
        if node.nxt: node.nxt.prv = node.prv

    def __insert(self, node: Node) -> None:
        node.prv = self.__recent.prv
        node.nxt = self.__recent
        if self.__recent.prv:
            self.__recent.prv.nxt = node
            self.__recent.prv = node
