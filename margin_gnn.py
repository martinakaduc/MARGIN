import networkx as nx
import numpy as np
import random
from data_structure import Cut
from typing import List, Dict, Tuple
from gnn_inference import InferenceGNN

def node_match(first_node, second_node):
        return first_node["label"] == second_node["label"]

def edge_match(first_edge, second_edge):
        return first_edge["label"] == second_edge["label"]

class MARGIN_GNN:
    def __init__(self, gnn_args, support=0.75, sortrep=False, randwalk=False) -> None:
        self.G = None
        self.sortrep = sortrep
        self.randwalk = randwalk
        self.support = support
        self.support_count = None
        self.cut_stack = []
        self.hashmap = []
        self.biggest_cut = 0
        self.num_explored_cut = 0
        self.mini_idx = 0
        self.MF = {}
        self.edges_with_freq = {}

        # For random walk
        self.equal_energy = 0.8
        self.lower_energy = 0.5
        self.nextcut_ratio = 10

        self.confidence = gnn_args.confidence
        self.inference_gnn = InferenceGNN(gnn_args)

    def run(self, G: Dict[int, nx.Graph]) -> List[nx.Graph]:
        self.G = G
        self.support_count = int(self.support * len(self.G))
        if self.sortrep:
            self.build_list_freq_edges()

        for i in range(len(self.G)):
            print("=== GRAPH %d ===" % i)
            self.mini_idx = i
            self.MF[i] = []
            representative, represent_path = self.find_representative()

            if representative == None:
                continue

            initial_cut = Cut()
            if len(represent_path) > 0:
                initial_cut.set_cut(representative, represent_path[-1])
            else:
                initial_cut.set_cut(representative)

            self.hashmap = []
            if self.randwalk:
                self.expand_cut_random(initial_cut)
            else:
                self.expand_cut(initial_cut)

        final = []
        line = []
        for gid, list_el in self.MF.items():
            self.mini_idx = gid
            for each_el in list_el:
                cm_sg = self.edgelist_to_graph(each_el)
                num_line = cm_sg.number_of_nodes() + cm_sg.number_of_edges()
                final.append(cm_sg)
                line.append(num_line)

        print("Top 10:", sorted(range(len(line)), key=lambda i: line[i], reverse=True)[:10])
        return final

    def build_list_freq_edges(self):
        for gid, g in self.G.items():
            for edge in g.edges:
                s = g.nodes[edge[0]]["label"]
                t = g.nodes[edge[1]]["label"]
                edge_fp = ""
                if s < t:
                    edge_fp = "%d-%d-%d" % (s, g[edge[0]][edge[1]]["label"], t)
                else:
                    edge_fp = "%d-%d-%d" % (t, g[edge[0]][edge[1]]["label"], s)
                
                if edge_fp in self.edges_with_freq:
                    self.edges_with_freq[edge_fp] += 1
                else:
                    self.edges_with_freq[edge_fp] = 1

    def smallest_graph(self) -> int:
        list_num_edge = [(i, g.number_of_edges()) for i, g in self.G.items()]
        min_edge_graph = min(list_num_edge, key=lambda x: x[1])

        return min_edge_graph[0]

    def support_graph(self, sub: nx.Graph) -> int:
        list_subgraphs = [sub] * len(self.G)
        list_graphs = list(self.G.values())
        labels = self.inference_gnn.predict_label(list_subgraphs, list_graphs)
        
        list_iso = list(filter(lambda x: x >= self.confidence, labels))
        return len(list_iso)

    def remove_edge(self, g: nx.Graph) -> Tuple[int]:
        random.seed(42)
        new_graph = None
        delete_edge = ()
        tried_edges = []
        orphan_node = None

        # if g.number_of_nodes() == g.number_of_edges() + 1:
        #     return None

        while True:
            delete_edge = random.choice(list(g.edges))
            while set(delete_edge) in tried_edges:
                delete_edge = random.choice(list(g.edges))
                # print(delete_edge)

            new_graph = g.copy()
            new_graph.remove_edge(*delete_edge)
            tried_edges.append(set(delete_edge))

            if nx.classes.function.is_empty(new_graph):
                delete_edge = ()
                break

            elif nx.is_connected(new_graph):
                break

            orphan_nodes = [nid for nid, degree in new_graph.degree if degree == 0]
            if len(orphan_nodes) == 1:
                orphan_node = orphan_nodes[0]
                break

        if len(delete_edge) > 0:
            g.remove_edge(*delete_edge)

        if orphan_node != None:
            g.remove_node(orphan_node)
            
        return delete_edge

    def remove_edge_order(self, g: nx.Graph, list_edges) -> Tuple[int]:
        new_graph = None
        delete_edge = ()
        orphan_node = None

        for i, delete_edge in enumerate(list_edges):
            new_graph = g.copy()
            new_graph.remove_edge(*delete_edge)

            if nx.classes.function.is_empty(new_graph):
                delete_edge = ()
                break

            elif nx.is_connected(new_graph):
                list_edges.pop(i)
                break

            orphan_nodes = [nid for nid, degree in new_graph.degree if degree == 0]
            if len(orphan_nodes) == 1:
                orphan_node = orphan_nodes[0]
                list_edges.pop(i)
                break

        if len(delete_edge) > 0:
            g.remove_edge(*delete_edge)

        if orphan_node != None:
            g.remove_node(orphan_node)
            
        return delete_edge

    def sort_edge_by_freq(self, g: nx.Graph) -> List[Tuple[int]]:
        edge_w_fp = []
        for edge in g.edges:
            s = g.nodes[edge[0]]["label"]
            t = g.nodes[edge[1]]["label"]
            edge_fp = ""
            if s < t:
                edge_fp = "%d-%d-%d" % (s, g[edge[0]][edge[1]]["label"], t)
            else:
                edge_fp = "%d-%d-%d" % (t, g[edge[0]][edge[1]]["label"], s)
            edge_w_fp.append(self.edges_with_freq[edge_fp])

        sorted_edges = list(sorted(zip(g.edges, edge_w_fp), key=lambda x: x[1]))
        return list(map(lambda x: x[0], sorted_edges))

    def find_representative(self) -> nx.Graph:
        print("Finding representative...")

        # self.mini_idx = self.smallest_graph()
        representative = self.G[self.mini_idx].copy()
        represent_path = []

        if self.sortrep:
            sorted_edges = self.sort_edge_by_freq(representative)
            while (self.support_graph(representative) < self.support_count):
                removed_edge = self.remove_edge_order(representative, sorted_edges)
                print(removed_edge)

                if len(removed_edge) == 0:
                    return None, None
                else:
                    represent_path.append(removed_edge)
        else:
            while (self.support_graph(representative) < self.support_count):
                removed_edge = self.remove_edge(representative)
                print(removed_edge)

                if len(removed_edge) == 0:
                    return None, None
                else:
                    represent_path.append(removed_edge)

        return representative, represent_path

    def set_hash_key(self, cut: Cut) -> None:
        hash_str = cut.get_hash_str()
        if hash_str not in self.hashmap:
            self.hashmap.append(hash_str)

    def get_hash_key(self, cut: Cut) -> bool:
        hash_str = cut.get_hash_str()
        return hash_str in self.hashmap

    def edgelist_to_graph(self, edgelist:List[Tuple[int]]) -> nx.Graph:
        graph = self.G[self.mini_idx].copy()

        removing_edges = set(graph.edges) - set(edgelist)
        graph.remove_edges_from(removing_edges)

        removing_nodes = [nid for nid, degree in graph.degree if degree == 0]
        graph.remove_nodes_from(removing_nodes)

        return graph

    def support_el(self, edgelist:List[Tuple[int]]) -> int:
        return self.support_graph(self.edgelist_to_graph(edgelist))

    def one_less_edge(self, cl:List[Tuple[int]]) -> List[List[Tuple[int]]]:
        list_parent = []
        for i in range(len(cl)):
            curr_cut = cl.copy()
            del curr_cut[i]
            p_graph = self.edgelist_to_graph(curr_cut)

            if not nx.classes.function.is_empty(p_graph) and nx.is_connected(p_graph):
                list_parent.append(curr_cut)

        return list_parent

    def one_more_edge(self, parrent: List[Tuple[int]]) -> List[List[Tuple[int]]]:
        list_children = []
        for edge in self.G[self.mini_idx].edges:
            if edge not in parrent:
                curr_cut = parrent.copy() + [edge]
                curr_cut = list(sorted(curr_cut))

                p_graph = self.edgelist_to_graph(curr_cut)

                if not nx.classes.function.is_empty(p_graph) and nx.is_connected(p_graph):
                    list_children.append(curr_cut)

        return list_children

    def find_common_child(self, cl1: List[Tuple[int]], cl2: List[Tuple[int]]) -> List[Tuple[int]]:
        common_child = set(cl1).union(set(cl2))
        common_child = list(sorted(common_child))
        return common_child
    
    def expand_cut(self, initial: Cut) -> None:
        print("Expanding cut...")
        self.cut_stack.append(initial)
        self.set_hash_key(initial)

        while len(self.cut_stack) > 0:
            curr_cut = self.cut_stack.pop(-1)

            if len(curr_cut.pl) > self.biggest_cut:
                self.biggest_cut = len(curr_cut.pl)

            self.num_explored_cut += 1
            all_parents = self.one_less_edge(curr_cut.cl)

            for parent in all_parents:
                if self.support_el(parent) >= self.support_count:
                    print(parent)
                    self.MF[self.mini_idx].append(parent)
                    all_children = self.one_more_edge(parent)

                    for child in all_children:
                        if self.support_el(child) < self.support_count:
                            add_cut = Cut()
                            add_cut.set(child, parent)
                            if not self.get_hash_key(add_cut):
                                self.set_hash_key(add_cut)
                                self.cut_stack.append(add_cut)
                        else:
                            cm_child = self.find_common_child(curr_cut.cl, child)
                            add_cut = Cut()
                            add_cut.set(cm_child, child)
                            if not self.get_hash_key(add_cut):
                                self.set_hash_key(add_cut)
                                self.cut_stack.append(add_cut)
                else:
                    grand_parents = self.one_less_edge(parent)
                    for gp in grand_parents:
                        if self.support_el(gp) >= self.support_count:
                            add_cut = Cut()
                            add_cut.set(parent, gp)
                            if not self.get_hash_key(add_cut):
                                self.set_hash_key(add_cut)
                                self.cut_stack.append(add_cut)

                            break

    def expand_cut_random(self, initial: Cut) -> None:
        print("Expanding cut...")
        self.cut_stack.append(initial)
        self.set_hash_key(initial)

        while len(self.cut_stack) > 0:
            curr_cut = self.cut_stack.pop(-1)
            print(curr_cut.pl)
            self.MF[self.mini_idx].append(curr_cut.pl)

            if len(curr_cut.pl) > self.biggest_cut:
                self.biggest_cut = len(curr_cut.pl)

            self.num_explored_cut += 1

            add_cut = self.get_next_cut(curr_cut)
            if not self.get_hash_key(add_cut):
                self.set_hash_key(add_cut)
                self.cut_stack.append(add_cut)
                size_cl = len(add_cut.cl)
                size_pl = len(add_cut.pl)

                while size_cl == 0 or \
                    size_pl == 0 or \
                    self.metropolis_1(size_pl, len(curr_cut.pl)) == 0:

                    self.cut_stack.pop(-1)
                    add_cut = self.get_next_cut(curr_cut)

                    if not self.get_hash_key(add_cut):
                        self.set_hash_key(add_cut)
                        self.cut_stack.append(add_cut)
                        size_cl = len(add_cut.cl)
                        size_pl = len(add_cut.pl)
                    else:
                        break

    def get_next_cut(self, curr: Cut) -> Cut:
        cut_type = np.random.randint(0, self.nextcut_ratio)
        next_cut = None

        if cut_type == 0:
            next_cut = self.get_type_pall(curr)
        elif cut_type == 1:
            next_cut = self.get_type_call(curr)
        elif cut_type == 2:
            next_cut = self.get_type_m(curr)
        elif cut_type == 3:
            next_cut = self.get_type_s1(curr)
        else:
            next_cut = self.get_type_global(curr, cut_type - 3)

        return next_cut

    def random_one_less_edge(self, cut: Cut) -> None:
        while True:
            remove_idx = np.random.randint(0, len(cut.cl))
            cut.pl = cut.cl.copy()
            del cut.pl[remove_idx]

            p_graph = self.edgelist_to_graph(cut.pl)

            if not nx.classes.function.is_empty(p_graph) and nx.is_connected(p_graph):
                break

    def random_one_more_edge(self, cut: Cut) -> None:
        list_cand_edge = list(set(self.G[self.mini_idx].edges) - set(cut.pl))
        while True:
            chose_edge = random.choice(list_cand_edge)
            cut.cl = cut.pl.copy() + [chose_edge]
            cut.cl = list(sorted(cut.cl))
            p_graph = self.edgelist_to_graph(cut.cl)

            if not nx.classes.function.is_empty(p_graph) and nx.is_connected(p_graph):
                break

    def get_type_pall(self, curr: Cut) -> Cut:
        if len(curr.cl) == 0:
            return Cut()

        final_cut = Cut()
        final_cut.cl = curr.cl.copy()
        self.random_one_less_edge(final_cut)
        count = 0

        while self.support_el(final_cut.pl) < self.support_count and \
              count <= self.G[self.mini_idx].number_of_edges():
            
            count += 1
            final_cut.cl = curr.cl.copy()
            self.random_one_less_edge(final_cut)

        if count > self.G[self.mini_idx].number_of_edges():
            return Cut()

        return final_cut

    def get_type_call(self, curr: Cut) -> Cut:
        pall = self.get_type_pall(curr)

        if len(pall.pl) == 0:
            return Cut()

        final_cut = Cut()
        final_cut.pl = pall.pl.copy()
        self.random_one_more_edge(final_cut)
        count = 0

        while self.support_el(final_cut.cl) >= self.support_count and \
              count <= self.G[self.mini_idx].number_of_edges():
            count += 1
            pall = self.get_type_pall(curr)
            final_cut.pl = pall.pl.copy()
            self.random_one_more_edge(final_cut)

        if count > self.G[self.mini_idx].number_of_edges():
            return Cut()

        return final_cut 

    def get_type_m(self, curr: Cut) -> Cut:
        pall = self.get_type_pall(curr)
        if len(pall.pl) == 0:
            return Cut()

        final_cut = Cut()
        final_cut.pl = pall.pl.copy()
        self.random_one_more_edge(final_cut)
        count = 0

        while self.support_el(final_cut.cl) < self.support_count and \
              count <= self.G[self.mini_idx].number_of_edges():

            count += 1
            final_cut.pl = pall.pl.copy()
            self.random_one_more_edge(final_cut)

        if count > self.G[self.mini_idx].number_of_edges():
            return Cut()
        
        cm_child = self.find_common_child(final_cut.cl, curr.cl)
        final_cut.pl = final_cut.cl
        final_cut.cl = cm_child

        return final_cut

    def get_type_s1(self, curr: Cut) -> Cut:
        if len(curr.cl) == 0:
            return Cut()

        final_cut = Cut()
        final_cut.cl = curr.cl.copy()
        self.random_one_less_edge(final_cut)
        count = 0

        while self.support_el(final_cut.pl) >= self.support_count and \
              count <= self.G[self.mini_idx].number_of_edges():

            count += 1
            final_cut.cl = curr.cl.copy()
            self.random_one_less_edge(final_cut)

        if count > self.G[self.mini_idx].number_of_edges():
            return Cut()

        final_cut.cl = final_cut.pl.copy()
        grand_parents = self.one_less_edge(final_cut.cl)
        for gp in grand_parents:
            if self.support_el(gp) >= self.support_count:
                final_cut.pl = gp
                break

        if len(final_cut.pl) == 0:
            return Cut()

        return final_cut

    def get_type_global(self, curr: Cut, change_time: int) -> Cut:
        final_cut = Cut()
        final_cut.set(curr.cl.copy(), curr.pl.copy())

        self.random_replace_edge(final_cut, change_time)

        return final_cut

    def random_replace_edge(self, cut: Cut, change_time: int) -> None:
        final_cut = Cut()
        final_cut.pl = cut.pl.copy()
        list_cand_edge = list(set(self.G[self.mini_idx].edges) - set(cut.pl))

        for _ in range(change_time):
            step_cut = final_cut.copy()
            chose_edge = None
            p_graph = nx.Graph()

            while len(list_cand_edge) > 0:
                chose_edge = random.choice(list_cand_edge)
                final_cut.pl = step_cut.pl.copy() + [chose_edge]
                final_cut.pl = list(sorted(final_cut.pl))
                p_graph = self.edgelist_to_graph(final_cut.pl)

                if nx.is_connected(p_graph):
                    break

            if not nx.classes.function.is_empty(p_graph) and self.support_graph(p_graph) >= self.support_count:
                list_cand_edge.remove(chose_edge)
            else:
                final_cut.pl = step_cut.pl.copy()
                break

        final_cut.cl = final_cut.pl
        step_cl = final_cut.cl.copy()
        while len(list_cand_edge) > 0:
            chose_edge = random.choice(list_cand_edge)
            final_cut.cl = step_cl + [chose_edge]
            final_cut.cl = list(sorted(final_cut.cl))
            p_graph = self.edgelist_to_graph(final_cut.cl)

            if nx.is_connected(p_graph):
                list_cand_edge.remove(chose_edge)
                step_cl = final_cut.cl.copy()

                if self.support_graph(p_graph) < self.support_count:
                    break

        cut.cl = final_cut.cl
        cut.pl = final_cut.pl

    def metropolis_1(self, enew: int, eold: int) -> int:
        if enew > eold:
            return 1
        elif enew < eold:
            r_num = np.random.uniform(0, 1)
            if r_num <= self.lower_energy:
                return 1
            return 0

        else:
            r_num = np.random.uniform(0, 1)
            if r_num <= self.equal_energy:
                return 1
            return 0