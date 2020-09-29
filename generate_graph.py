import numpy as np
import random

def generate(num_of_graphs, min_node, max_node, subgraph_size, edge_fill=0.75):
    graph_db = []
    graph_index = []
    subgraph = np.zeros((subgraph_size, subgraph_size), dtype=np.int)

    # Random filling index
    for _ in range(num_of_graphs):
        length = random.randint(min_node, max_node)
        graph_index.append(random.sample(range(length), k=length))

    # Random subgraph
    for i in range(subgraph_size):
        node_val = random.randint(1, 500)
        subgraph[i][i] = node_val

    subgraph_total_edge = sum(range(subgraph_size))

    for _ in range(int(edge_fill*subgraph_total_edge)):
        y, x = np.where(subgraph == 0)
        y, x = random.choice(list(zip(y, x)))

        edge_val = random.randint(1, 500)
        subgraph[x][y] = edge_val
        subgraph[y][x] = edge_val

    # Intergrate subgraph and random graph
    for i in range(num_of_graphs):
        # Copy subgraph
        length = len(graph_index[i])
        graph = np.zeros((length,length), dtype=np.int)

        for k in range(subgraph_size):
            graph[graph_index[i][k]][graph_index[i][k]] = subgraph[k][k] # Node
            iter_j = np.where(subgraph[k] > 0)[0]
            iter_j = iter_j[iter_j > k]
            # Edge
            for j in iter_j:
                graph[graph_index[i][k]][graph_index[i][j]] = subgraph[k][j]
                graph[graph_index[i][j]][graph_index[i][k]] = subgraph[j][k]

        # Generate random remaining node & edge
        for k in graph_index[i][subgraph_size:]:
            node_val = random.randint(1, 500)
            graph[k][k] = node_val

        graph_total_edge = sum(range(length))

        for _ in range(int(edge_fill*(graph_total_edge-subgraph_total_edge))):
            y, x = np.where(graph == 0)
            y, x = random.choice(list(zip(y, x)))

            edge_val = random.randint(1, 500)
            graph[x][y] = edge_val
            graph[y][x] = edge_val

            x = x + 1
            if x == length:
                x = subgraph_size

        for k in range(length):
            if np.sum(graph[k] > 0) <= 1:
                # Orphan node =>> need remove
                graph = np.delete(graph, k, axis=0)
                graph = np.delete(graph, k, axis=1)

        graph_db.append(graph)

    return np.asarray(graph_db), np.asarray([x[:subgraph_size] for x in graph_index])

if __name__ == '__main__':
    graph_db, _ = generate(num_of_graphs=10, min_node=8, max_node=10, subgraph_size=5)
    for x in graph_db:
        print(x)
    print(_)
