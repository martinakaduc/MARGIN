import numpy as np
from generate_graph import generate
from margin import Graph, GraphCollection, encodeGraph
from utils import *

if __name__ == '__main__':
    datasets = "8graphs_2pattern15nodes.lg"
    min_subgraph = 17 # Số node của frequent subgraph
    min_support = 0.4
    graphs = []

    # graph_input = []
    # graph_input.append(np.array([[  1,  44,  44,  67,   0,   0,   0,   0],
    #        [ 44,   3,  17,  95,  13,  50, 105,  58],
    #        [ 44,  17,  28,   0,  44,   0,   33,  52],
    #        [ 67,  95,   0,  25,   0,  78,   0,   0],
    #        [  0,  13,   44,  0,  30,   0,  54,  98],
    #        [  0,  50,   0,  78,   0,  22,   0,   0],
    #        [  0, 105,   33,   0,  54,   0,  27,   0],
    #        [  0,  58,  52,   0,  98,   0,   0,  14]]))
    #
    # graph_input.append(np.array([[  3,  13,  17,  95,  50, 105,  44,  58],
    #        [ 13,  30,   48,   0,   59,  54,   0,  98],
    #        [ 17,   48,  28,   0,   0,   0,  44,   0],
    #        [ 95,   0,   0,  25,   0,   0,  67,   0],
    #        [ 50,   59,   0,   0,  22,   0,   0,   15],
    #        [105,  54,   0,   0,   0,  27,   4,   0],
    #        [ 44,   0,  44,  67,   0,   4,   1,   0],
    #        [ 58,  98,   0,   0,   15,   0,   0,  14]]))
    #
    # graph_input.append(np.array([[  3,  13,  95,  50, 105,  44,  58,  17],
    #        [ 13,  30,   0,   12,  54,   0,  98,   48],
    #        [ 95,   0,  25,   0,   0,  64,   0,   21],
    #        [ 50,   12,   0,  22,   0,   56,   0,   0],
    #        [105,  54,   0,   0,  27,   0,   0,   0],
    #        [ 44,   0,  64,   56,   0,   1,   0,  44],
    #        [ 58,  98,   0,   0,   0,   0,  14,   0],
    #        [ 17,   48,   21,   0,   0,  44,   0,  28]]))

    graph_input = readGraphs(datasets)
    # graph_input, _ = generate(num_of_graphs=3, min_node=5, max_node=7, subgraph_size=4, edge_fill=0.6)
    # plotGraph(graph_input[0], False)
    # plotGraph(graph_input[1], False)
    # plotGraph(graph_input[2], False)

    # print("Graph 0: ", encodeGraph(graph_input[0]))
    # print("Graph 1: ", encodeGraph(graph_input[1]))
    # print("Graph 2: ", encodeGraph(graph_input[2]))

    for i, graph_array in enumerate(graph_input):
        print(graph_array)
        print("CONSTRUCTING LATTICE SEARCH SPACE... Graph %d" % i)
        graphs.append(Graph(graph_array, min_subgraph=min_subgraph))
        print("Graph %d has lattice space length %d." % (i, len(graphs[i].lattice["code"])))

    # print(graphs[0].lattice["code"])
    # print(graphs[1].lattice["code"])
    # print(graphs[2].lattice["code"])
    graphDB = GraphCollection(graphs, min_support)
    MF = graphDB.margin()

    print(MF)
    for sg in MF["tree"]:
        plotGraph(sg, False)
