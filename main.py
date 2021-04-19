import numpy as np
from generate_graph import generate
from margin import Graph, GraphCollection, encodeGraph
from utils import *
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--min_support',
        type=int,
        default=5000,
        help='min support, default 5000'
    )
    parser.add_argument(
        '-e', '--min_edge',
        type=int,
        default=10,
        help='min edge, default 10'
    )
    parser.add_argument(
        'database_file_name',
        type=str,
        help='str, database file name'
    )

    args = parser.parse_args()
    graphs = []
    graph_input = readGraphs(args.database_file_name)
    start_time = time.time()

    for i, graph_array in enumerate(graph_input):
        print(graph_array)
        print("CONSTRUCTING LATTICE SEARCH SPACE... Graph %d" % i)
        graphs.append(Graph(graph_array, min_edge=args.min_edge))
        print("Graph %d has lattice space length %d." % (i, len(graphs[i].lattice["code"])))

    graphDB = GraphCollection(graphs, args.min_support)
    MF = graphDB.margin()

    print(MF)
    print("RUNNING TIME: %.5f s"%(time.time() - start_time))
    # for sg in MF["tree"]:
    #     plotGraph(sg, False)
