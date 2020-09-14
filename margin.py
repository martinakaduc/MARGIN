import numpy as np
import copy

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
    def __init__(self, graph_):
        self.data = np.array(graph_)
        self.lattice = {"tree": [], "code": [], "children": [], "parents": []}
        self.generateLatticeSpace(self.data) # {"tree": [list of subgraph], "code"["list of subgraph embed"]}
        self.frequent_lattice = [-1] * len(self.lattice["tree"])

    def generateLatticeSpace(self, graph_, child=-1):
        # Generate all subgraphs
        # Return: {"tree": [list of subgraph], "code"["list of subgraph embed"],
        #          "children": [list of children index], "parents": [list of parrent index]}
        # List in decrease order
        # TODO
        if graph_.shape[0] == 0:
            return

        tempGraph = copy.deepcopy(graph_)

        # Add current node (subgraph) to lattice space
        embed = embedGraph(tempGraph)
        if embed["code"] not in self.lattice["code"]:
            self.lattice["tree"].append(embed["tree"])
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


        num_edge = np.sum(tempGraph > 0, axis=0)
        traverse_order = np.argsort(num_edge)

        while np.sum(traverse_order > 0) > 0:
            # Find the most-edge node
            node_i = np.where(traverse_order==0)[0][0]
            # Drop an edge and ensure the remaining graph is connected
            for edge_i in range(tempGraph.shape[0]):
                if edge_i == node_i and tempGraph.shape[0] > 1:
                    continue

                if tempGraph[node_i][edge_i] == 0:
                    continue

                drop_success = False
                dropGraph = copy.deepcopy(tempGraph)
                dropGraph[node_i][edge_i] = 0
                dropGraph[edge_i][node_i] = 0

                if isGraphConnected(dropGraph):
                    drop_success = True
                else:
                    for run_i in list(sorted([node_i, edge_i], reverse=True)):
                        if np.sum(dropGraph[run_i] > 0) <= 1:
                            # Orphan node =>> need remove
                            dropGraph = np.delete(dropGraph, run_i, axis=0)
                            dropGraph = np.delete(dropGraph, run_i, axis=1)

                            drop_success = True

                if drop_success:
                    self.generateLatticeSpace(graph_=dropGraph, child=self.lattice["code"].index(embed["code"]))

            traverse_order -= 1

        # Write parents from children
        for i, children in enumerate(self.lattice["children"]):
            for child_i in children:
                if i not in self.lattice["parents"][child_i]:
                    self.lattice["parents"][child_i].append(i)

    def haveSubgraph(self, subgraph):
        return subgraph in self.lattice["code"]

class GraphCollection():
    def __init__(self, graphs_, theta_):
        self.graphs = graphs_
        self.theta = int(theta_ * len(graphs_))
        self.length = len(graphs_)

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
        for i, subgraph in enumerate(target_graph.lattice["code"]):
            if self.isFrequent(subgraph):
                # Found representative
                target_graph.frequent_lattice[i] = True
                return i
            else:
                target_graph.frequent_lattice[i] = False

        return -1

    def expandCut(self, Gi, LF, cut, _cut_visited=[]):
        C, P = cut
        cut_visited = copy.deepcopy(_cut_visited)
        cut_visited.append(cut)
        Y_list = Gi.lattice["parents"][C] # Index of C parrents in graph.lattice

        for Yi in Y_list:
            # Check Yi is frequent
            is_frequent = self.checkGraphLatticeFrequent(Gi, Yi)
            # print(Gi.lattice["code"][Yi])
            # print(self.checkGraphLatticeFrequent(Gi, Yi))
            if is_frequent:
                LF.append([Gi.lattice["code"][Yi], Gi.lattice["tree"][Yi]])
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
                                LF, cut_visited = self.expandCut(Gi, LF, (M_list[0], K), cut_visited)
                                break

                    else: # K is infrequent
                        if (K, Yi) not in cut_visited:
                            LF, cut_visited = self.expandCut(Gi, LF, (K, Yi), cut_visited)

            else:  # Yi is infrequent
                Y_parents = Gi.lattice["parents"][Yi]
                for Y_p in Y_parents:
                    if self.checkGraphLatticeFrequent(Gi, Y_p) and (Yi, Y_p) not in cut_visited:
                        # print(Gi.lattice["code"][Y_p])
                        # print(Gi.lattice["code"][Yi])
                        LF, cut_visited = self.expandCut(Gi, LF, (Yi, Y_p), cut_visited)
                        break

        return LF, cut_visited


    def merge(self, MF, LF):
        if len(MF["code"]) == 0:
            length_list = [len(x[0]) for x in LF]
            if length_list:
                max_len = max(length_list)
                # Filter only the longest subgraph
                return {"tree": [x[1] for x in LF if len(x[0]) == max_len],
                        "code": [x[0] for x in LF if len(x[0]) == max_len]}
            else:
                return MF

        else:
            max_len_MF = len(MF["code"][0])

            length_list = [len(x[0]) for x in LF]
            max_len_LF = max(length_list)
            # Decide what to remove & what is the maximum subgraphs
            if max_len_MF > max_len_LF:
                return MF
            elif max_len_MF < max_len_LF:
                return {"tree": [x[1] for x in LF if len(x[0]) == max_len],
                        "code": [x[0] for x in LF if len(x[0]) == max_len]}
            else:
                for co, tr in LF:
                    if co not in MF["code"] and len(co) == max_len_MF:
                        MF["tree"].append(tr)
                        MF["code"].append(co)
                return MF

    def margin(self):
        MF = {"tree": [], "code": []}

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
            LF.append([Gi.lattice["code"][Ri], Gi.lattice["tree"][Ri]])

            # Expand cut
            print("SPANNING...")
            CRi_list = [x for x in Gi.lattice["children"][Ri] if Gi.frequent_lattice[x] == False]
            if CRi_list:
                LF, _ = self.expandCut(Gi, LF, (CRi_list[0], Ri), [])
            print("LF: ", LF)

            # Merfe MF and LF
            print("MERGING...")
            MF = self.merge(MF, LF)

        return MF
