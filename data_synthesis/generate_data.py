import os
import json
import argparse
import numpy as np
import networkx as nx

from tqdm import tqdm
from multiprocessing import Process

def parse_args():
    parser = argparse.ArgumentParser(description='Synthetic graphs')
    parser.add_argument("--config", "-c", default="configs/base.json", type=str, help="Config file")
    parser.add_argument("--cont", default=False, type=bool, help="Continue generating")
    return parser.parse_args()

def read_config(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def add_labels(graph, NN, NE):
    nodes = np.array(list(graph.nodes))
    edges = np.array(list(graph.edges))

    node_labels = np.random.randint(0, NN, len(nodes)).tolist()
    edge_labels = np.random.randint(0, NE, len(edges)).tolist()

    labelled_nodes = [(nodes[k], {'label': node_labels[k], 'color': 'green'}) for k in range(len(nodes))]
    labelled_edges = [(edges[k][0], edges[k][1], {'label': edge_labels[k], 'color': 'green'}) for k in range(len(edges))]

    G = nx.Graph()
    G.add_nodes_from(labelled_nodes)
    G.add_edges_from(labelled_edges)

    return G

def generate_patterns(number_of_pattern, avg_pattern_size, std_pattern_size,
                      avg_degree, std_degree, number_label_node, number_label_edge,
                      *args, **kwargs):
    list_patterns = []
    for i in range(number_of_pattern):
        generated_pattern = None
        iteration = 0
        no_of_nodes = int(np.random.normal(avg_pattern_size, std_pattern_size))
        degree = np.random.normal(avg_degree, std_degree)
        probability_for_edge_creation = degree / (no_of_nodes - 1)

        while generated_pattern is None or not nx.is_connected(
                    generated_pattern):  # make sure the generated graph is connected
            generated_pattern = nx.erdos_renyi_graph(no_of_nodes, probability_for_edge_creation, directed=False)
            iteration += 1
            if iteration > 5:
                probability_for_edge_creation *= 1.05
                iteration = 0

        labelled_pattern = add_labels(generated_pattern, number_label_node, number_label_edge)
        list_patterns.append(labelled_pattern)

    return list_patterns
def add_random_edges(current_graph, NE, min_edge, max_edge):
    """
    randomly adds edges between nodes with no existing edges.
    based on: https://stackoverflow.com/questions/42591549/add-and-delete-a-random-edge-in-networkx
    :param probability_of_new_connection:
    :return: None
    """
    if current_graph:
        new_edges = []
        connected = []
        for i in range(len(current_graph.nodes())):
            # find the other nodes this one is connected to
            connected = connected + [to for (fr, to) in current_graph.edges(i)]
            connected = list(dict.fromkeys(connected))
            # and find the remainder of nodes, which are candidates for new edges

        unconnected = [j for j in range(len(current_graph.nodes())) if not j in connected]
        # print('Connected:', connected)
        # print('Unconnected', unconnected)
        is_connected = False
        while not is_connected:  # randomly add edges until the graph is connected
            if len(unconnected)==0:
                break
            new = np.random.choice(unconnected)
            edge_label = np.random.randint(0, NE)

            # for visualise only
            current_graph.add_edges_from([(np.random.choice(connected), new, {'label': edge_label})])
            # book-keeping, in case both add and remove done in same cycle
            unconnected.remove(new)
            connected.append(new)
            is_connected = nx.is_connected(current_graph)
            # print('Connected:', connected)
            # print('Unconnected', unconnected

        num_edges = np.random.randint(min_edge, max_edge+1)

        while current_graph.number_of_edges() < num_edges:
            edge_label = np.random.randint(0, NE)
            current_graph.add_edges_from([(np.random.choice(connected), np.random.choice(connected), {'label': edge_label})])

    return current_graph

def gen_transaction(pattern, avg_graph_size, std_graph_size,
                     avg_degree, std_degree, number_label_node, number_label_edge,
                     *args, **kwargs):
    no_of_nodes = int(np.random.normal(avg_graph_size, std_graph_size))
    pattern_nodes = pattern.number_of_nodes()
    while no_of_nodes < pattern_nodes:
        no_of_nodes = int(np.random.normal(avg_graph_size, std_graph_size))

    node_id = pattern_nodes  # start node_id from the number of nodes already in the common graph (note that the node ids are numbered from 0)
    added_nodes = []
    for _ in range(no_of_nodes - pattern_nodes):
        node_label = np.random.randint(0, number_label_node)
        added_nodes.append((node_id, {'label': node_label}))
        node_id += 1
    pattern.add_nodes_from(added_nodes)

    min_edges = int((avg_degree - std_degree) * node_id / 2)
    max_edges = int((avg_degree + std_degree) * node_id / 2)
    
    add_random_edges(pattern, number_label_edge, min_edges, max_edges)

    return pattern

def gen_transactions(patterns, number_of_graph, *args, **kwargs):
    list_graphs = []
    for i in tqdm(range(number_of_graph)):
        labelled_graph = None
        injected_pattern_idx = np.random.randint(0, len(patterns))
        injected_pattern = patterns[injected_pattern_idx].copy()
        labelled_graph = gen_transaction(injected_pattern, *args, **kwargs)
        list_graphs.append(labelled_graph)
    return list_graphs

def save_attributed_graphs(graphs, saved_file):
    # Save graphs
    with open(saved_file, 'w', encoding='utf-8') as file:
        for i, graph in enumerate(graphs):
            file.write('t # {0}\n'.format(i))
            for node in graph.nodes:
                file.write('v {} {}\n'.format(node, graph.nodes[node]['label']))
            for edge in graph.edges:
                file.write('e {} {} {}\n'.format(edge[0], edge[1], graph.edges[(edge[0], edge[1])]['label']))

def generate_dataset(dataset_path, is_continue, *args, **kwargs):
    print("Generating...")

    patterns = generate_patterns(*args, **kwargs)
    pattern_file = os.path.join(dataset_path, "patterns.lg")
    save_attributed_graphs(patterns, pattern_file)

    graphs = gen_transactions(patterns, *args, **kwargs)
    graph_file = os.path.join(dataset_path, "graphs.lg")
    save_attributed_graphs(graphs, graph_file)

def main(config_file, is_continue):
    np.random.seed(42)
    dataset_path = os.path.join("../datasets", 
                   os.path.basename(config_file).split(".")[0])
    ensure_path(dataset_path)
    config = read_config(config_file)

    generate_dataset(dataset_path=dataset_path, is_continue=is_continue, **config)

if __name__ == '__main__':
    args = parse_args()
    main(args.config, args.cont)
