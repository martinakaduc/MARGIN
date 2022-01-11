import networkx as nx
from typing import List, Tuple

class Cut:
    def __init__(self) -> None:
        self.cl = []
        self.pl = []

    def set_cut(self, g: nx.Graph, edge_index=None) -> None:
        self.pl = list(sorted(list(g.edges)))

        if edge_index != None:
            self.cl = self.pl.copy() + [edge_index]
            self.cl = list(sorted(self.cl))

    def get_hash_str(self) -> str:
        hash_str = ",".join(["%d-%d"%x for x in self.cl])
        hash_str += "@"
        hash_str += ",".join(["%d-%d"%x for x in self.pl])
        return hash_str

    def set(self, cl:List[Tuple[int]], pl:List[Tuple[int]]) -> None:
        self.cl = cl
        self.pl = pl

    def copy(self):
        new = Cut()
        new.set(self.cl.copy(), self.pl.copy())
        return new