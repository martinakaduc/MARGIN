import numpy as np
import copy
import collections
import uuid

def encodeGraph(graph):
    visited = [False]*len(graph)
    labelNodes = graph.diagonal()
    startNode = np.argmax(labelNodes)

    queue = []
    queue.append(startNode)
    code = str(graph[startNode,startNode]) + '$'

    while queue:
        s = queue.pop(0)
        visited[s] = True
        levelStr = ''

        edge_list = np.where(graph[s]>0)[0].tolist()
        # Sort edge
        node_list = [graph[x, x] for x in edge_list]

        edge_list = list(sorted(zip(edge_list, node_list), key=lambda x: x[1], reverse=True))
        # print(edge_list)

        for i, _ in edge_list:
            if not visited[i]:
                if i not in queue:
                    queue.append(i)

            if s != i:
                levelStr += str(graph[s,i]) + "_" + str(graph[i,i]) + "_"
                # visited[i] = True

        if levelStr != '':
            code += levelStr[:-1] +  '$'

    code += '#'

    return code

def embedGraph(graph):
    return {"tree": graph, "code": encodeGraph(graph)}

def isGraphConnected(_graph):
    graph = copy.deepcopy(_graph)

    queue = []
    queue.append(0)
    visited = []

    while queue:
        currentNode = queue.pop(0)
        if currentNode not in visited:
            visited.append(currentNode)
        else:
            continue
        row = np.where(graph[currentNode] > 0)[0]
        for e in row:
            if e != currentNode and e not in visited:
                queue.append(e)
                graph[currentNode][e] = 0
                graph[e][currentNode] = 0

    return len(visited) == len(graph)

class Graph():
    def __init__(self, graph_, min_edge=2):
        self.data = np.array(graph_)
        self.lattice = {"tree": [], "code": [], "children": [], "parents": []}
        self.generateLatticeSpace(self.data, min_edge=min_edge) # {"tree": [list of subgraph], "code"["list of subgraph embed"]}
        self.writeParrents()
        self.frequent_lattice = [-1] * len(self.lattice["tree"])

    def generateLatticeSpace(self, tempGraph, min_edge=2, child=-1):
        # Generate all subgraphs
        # Return: {"tree": [list of subgraph], "code"["list of subgraph embed"],
        #          "children": [list of children index], "parents": [list of parrent index]}
        # List in decrease order
        # TODO

        # Add current node (subgraph) to lattice space
        embed = embedGraph(tempGraph)
        # print(embed["code"])
        if embed["code"] not in self.lattice["code"]:
            self.lattice["tree"].append(embed["tree"].copy())
            self.lattice["code"].append(embed["code"])
            if child == -1:
                self.lattice["children"].append([])
            else:
                self.lattice["children"].append([child])
            self.lattice["parents"].append([])
        else:
            lattice_i = self.lattice["code"].index(embed["code"])
            if child not in self.lattice["children"][lattice_i]:
                self.lattice["children"][lattice_i].append(child)
            return

        if (np.sum(tempGraph > 0) - tempGraph.shape[0]) / 2 <= min_edge:
            return

        num_edge = np.sum(tempGraph > 0, axis=0)
        traverse_order = np.argsort(num_edge).tolist()[::-1]
        # print(traverse_order)

        while traverse_order:
            # Find the most-edge node
            node_i = traverse_order.pop()
            # Drop an edge and ensure the remaining graph is connected
            for edge_i in range(tempGraph.shape[0]):
                if edge_i == node_i and tempGraph.shape[0] > 1:
                    continue

                if tempGraph[node_i][edge_i] == 0:
                    continue

                drop_success = False

                backup_edge_label = tempGraph[node_i][edge_i]
                backup_node_index = []
                backup_node_label = []

                tempGraph[node_i][edge_i] = 0
                tempGraph[edge_i][node_i] = 0

                if isGraphConnected(tempGraph):
                    drop_success = True
                else:
                    if (np.sum(tempGraph[node_i] > 0) <= 1 and np.sum(tempGraph[edge_i] > 0) > 1) or \
                       (np.sum(tempGraph[node_i] > 0) > 1 and np.sum(tempGraph[edge_i] > 0) <= 1):
                        for run_i in list(sorted([node_i, edge_i], reverse=True)):
                            if np.sum(tempGraph[run_i] > 0) <= 1:
                                # Orphan node =>> need remove
                                backup_node_index.append(run_i)
                                backup_node_label.append(tempGraph[run_i][run_i])

                                tempGraph = np.delete(tempGraph, run_i, axis=0)
                                tempGraph = np.delete(tempGraph, run_i, axis=1)

                                drop_success = True
                                break

                if drop_success:
                    # cur_freq = GraphCollection.checkFreq(embed["code"])
                    # if meet_freq and cur_freq:
                    #     continue

                    self.generateLatticeSpace(tempGraph,
                                              min_edge=min_edge,
                                              child=self.lattice["code"].index(embed["code"]))

                # if len(backup_node_index) > 1:
                #     print("FUCKKIKIVJJOIRJCVOIJRNEO %d" % len(backup_node_index))

                while len(backup_node_index) > 0:
                    ni = backup_node_index.pop()
                    nl = backup_node_label.pop()
                    tempGraph = np.insert(tempGraph, ni, 0, axis=1)
                    tempGraph = np.insert(tempGraph, ni, 0, axis=0)
                    tempGraph[ni][ni] = nl


                tempGraph[node_i][edge_i] = backup_edge_label
                tempGraph[edge_i][node_i] = backup_edge_label


    def writeParrents(self):
        # Write parents from children
        for i, children in enumerate(self.lattice["children"]):
            for child_i in children:
                if i not in self.lattice["parents"][child_i]:
                    self.lattice["parents"][child_i].append(i)

    def haveSubgraph(self, subgraph):
        return subgraph in self.lattice["code"]

    def set_freq_edge(vid1, vid2, freq):
        pass

class GraphCollection():
    def __init__(self, graphs_, theta_):
        self.graphs = graphs_
        self.theta = theta_
        self.length = len(graphs_)
        # self._frequent_edges  = []
        # self.findFreqEdges()
        # print(self._frequent_edges)

    def findFreqEdges(self):
        vevlb_counter = collections.Counter()
        vevlb_counted = set()
        vevlb_dict = dict()

        for gid, g in enumerate(self.graphs):
            for vid1 in range(g.data.shape[0]):
                list_to = np.where(g.data[vid1, vid1+1:] > 0)[0] + vid1 + 1
                for vid2 in list_to:
                    vlb1, vlb2 = g.data[vid1][vid1], g.data[vid2][vid2]
                    elb = g.data[vid1][vid2]

                    if vlb1 < vlb2:
                        vlb1, vlb2 = vlb2, vlb1
                        vid1, vid2 = vid2, vid1

                    if (gid, (vlb1, elb, vlb2)) not in vevlb_counted:
                        vevlb_counter[(vlb1, elb, vlb2)] += 1
                    vevlb_counted.add((gid, (vlb1, elb, vlb2)))

                    if (vlb1, elb, vlb2) not in vevlb_dict:
                        vevlb_dict[(vlb1, elb, vlb2)] = {}

                    if gid not in vevlb_dict[(vlb1, elb, vlb2)]:
                        vevlb_dict[(vlb1, elb, vlb2)][gid] = []

                    if [vid1, vid2] not in vevlb_dict[(vlb1, elb, vlb2)][gid] and [vid2, vid1] not in vevlb_dict[(vlb1, elb, vlb2)][gid]:
                        vevlb_dict[(vlb1, elb, vlb2)][gid].append([vid1, vid2])

        self._frequent_edges = vevlb_dict

        for vevlb, cnt in vevlb_counter.items():
            if cnt >= self._min_support:
                # Mark edge as frequent
                for gid, g in enumerate(self.graphs):
                    if gid in vevlb_dict[vevlb]:
                        for pair in vevlb_dict[vevlb][gid]:
                            self.graphs[gid].set_freq_edge(pair[0], pair[1], list(vevlb_dict[vevlb].keys()))


    def sigma(self, subgraph, graph):
        return graph.haveSubgraph(subgraph)

    def isFrequent(self, subgraph):
        count = 0

        for graph in self.graphs:
            if self.sigma(subgraph, graph):
                count += 1

        return count >= self.theta

    def checkGraphLatticeFrequent(self, Gi, index):
        if Gi.frequent_lattice[index] == -1:
            if self.isFrequent(Gi.lattice["code"][index]):
                is_frequent = True
                Gi.frequent_lattice[index] = True
            else:
                is_frequent = False
                Gi.frequent_lattice[index] = False
        else:
            is_frequent = Gi.frequent_lattice[index]

        return is_frequent

    def findRepresentative(self, target_graph):
        if len(target_graph.lattice["parents"]) == 0:
            return -1

        queue = [0]
        # for i, subgraph in enumerate(target_graph.lattice["code"]):
        #     if self.isFrequent(subgraph):
        #         # Found representative
        #         target_graph.frequent_lattice[i] = True
        #         return i
        #     else:
        #         target_graph.frequent_lattice[i] = False
        visited = []
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.append(node)
            else:
                continue

            for p in target_graph.lattice["parents"][node]:
                if p not in visited:
                    queue.append(p)

            if self.isFrequent(target_graph.lattice["code"][node]):
                # Found representative
                target_graph.frequent_lattice[node] = True
                return node
            else:
                target_graph.frequent_lattice[node] = False

        return -1

    def expandCut(self, gid, Gi, LF, cut, cut_visited=[], lattice_node_visited=[]):
        C, P = cut
        # cut_visited = copy.deepcopy(_cut_visited)
        # lattice_node_visited = copy.deepcopy(_lattice_node_visited)

        cut_visited.append(cut)
        if C not in lattice_node_visited:
            lattice_node_visited.append(C)
        if P not in lattice_node_visited:
            lattice_node_visited.append(P)

        Y_list = Gi.lattice["parents"][C] # Index of C parrents in graph.lattice

        for Yi in Y_list:
            if Yi in lattice_node_visited:
                continue
            else:
                lattice_node_visited.append(Yi)

            # Check Yi is frequent
            is_frequent = self.checkGraphLatticeFrequent(Gi, Yi)
            # print(Gi.lattice["code"][Yi])
            # print(self.checkGraphLatticeFrequent(Gi, Yi))
            if is_frequent:
                LF.append([Gi.lattice["code"][Yi], Gi.lattice["tree"][Yi], gid])
                Y_child = Gi.lattice["children"][Yi]
                for K in Y_child:
                    # print(K, C)
                    if K == C:
                        continue
                    k_is_frequent = self.checkGraphLatticeFrequent(Gi, K)

                    if k_is_frequent: # K is frequent
                        # Find common child M of C and K
                        C_children = Gi.lattice["children"][C]
                        K_children = Gi.lattice["children"][K]
                        M_list = set(C_children) & set(K_children)
                        M_list = list(M_list)
                        # print(M_list[0])
                        for M in M_list:
                            if (M, K) not in cut_visited:
                                LF = self.expandCut(gid, Gi, LF, (M, K), cut_visited, lattice_node_visited)
                                break

                    else: # K is infrequent
                        if (K, Yi) not in cut_visited:
                            LF = self.expandCut(gid, Gi, LF, (K, Yi), cut_visited, lattice_node_visited)

            else:  # Yi is infrequent
                Y_parents = Gi.lattice["parents"][Yi]
                for Y_p in Y_parents:
                    if self.checkGraphLatticeFrequent(Gi, Y_p) and (Yi, Y_p) not in cut_visited:
                        # print(Gi.lattice["code"][Y_p])
                        # print(Gi.lattice["code"][Yi])
                        LF = self.expandCut(gid, Gi, LF, (Yi, Y_p), cut_visited, lattice_node_visited)
                        break

        return LF


    def merge(self, MF, LF):
        if len(MF["code"]) == 0:
            length_list = [len(x[0]) for x in LF]
            if length_list:
                max_len = max(length_list)
                # Filter only the longest subgraph
                for x in LF:
                    if len(x[0]) == max_len:
                        if x[0] not in MF["code"]:
                            MF["tree"].append(x[1])
                            MF["code"].append(x[0])
                            MF["freq"].append([x[2]])

        else:
            for co, tr, fe in LF:
                if co not in MF["code"]:
                    MF["tree"].append(tr)
                    MF["code"].append(co)
                    MF["freq"].append([fe])
                elif co in MF["code"]:
                    index = MF["code"].index(co)
                    MF["freq"][index].append(fe)

        return MF

    def margin(self):
        MF = {"tree": [], "code": [], "freq": []}

        for i, Gi in enumerate(self.graphs):
            print("RUNNING GRAPH NO. %d" % i)
            LF = []

            # Find the representative Ri of Gi
            print("FIND REPRESENTATIVE...")
            Ri = self.findRepresentative(Gi)
            if Ri == -1:
                continue
            print("Represent: ", Gi.lattice["code"][Ri])

            # Append the representative to LF
            LF.append([Gi.lattice["code"][Ri], Gi.lattice["tree"][Ri], i])

            # Expand cut
            print("SPANNING...")
            CRi_list = [x for x in Gi.lattice["children"][Ri] if Gi.frequent_lattice[x] == False]
            if CRi_list:
                LF = self.expandCut(i, Gi, LF, (CRi_list[0], Ri))
            print("LF: ", LF)

            # Merfe MF and LF
            print("MERGING...")
            MF = self.merge(MF, LF)

        for i, freq in enumerate(MF["freq"]):
            MF["freq"][i] = list(set(freq))

        # max_freq = max([len(x) for x in MF["freq"]])
        #
        # tobe_del = []
        # for i, freq in enumerate(MF["freq"]):
        #     if len(freq) < max_freq:
        #         tobe_del.append(i)
        #
        # for i in tobe_del[::-1]:
        #     del MF["code"][i]
        #     del MF["tree"][i]
        #     del MF["freq"][i]

        return MF
