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
        default=1,
        help='min edge, default 1'
    )
    parser.add_argument(
        '-m', '--max_size',
        type=int,
        default=35,
        help='max graph size'
    )
    parser.add_argument(
        '-t', '--max_time',
        type=int,
        default=86400, # 12hrs
        help='max time (second)'
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
        # print(graph_array)
        print("CONSTRUCTING LATTICE SEARCH SPACE... Graph %d" % i)
        graphs.append(Graph(graph_array, min_edge=args.min_edge, max_size=args.max_size, start_time=start_time, max_time=args.max_time))
        print("Graph %d has lattice space length %d." % (i, len(graphs[i].lattice["code"])))

    if time.time() - start_time < args.max_time:
        graphDB = GraphCollection(graphs, args.min_support)
        MF = graphDB.margin()

        print("RESULT")
        for i, code in enumerate(MF["code"]):
            print("Pattern %d" % i)
            print("Code:", MF["code"][i])
            print("Support:", MF["freq"][i])

    print("RUNNING TIME: %.5f s"%(time.time() - start_time))

    # max_pattern = max(len(x) for x in MF["tree"])
    # print(max_pattern)
    # for sg in MF["tree"]:
    #     plotGraph(sg, False)
