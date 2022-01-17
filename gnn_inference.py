import os
import utils
import torch
import argparse
import numpy as np
import networkx as nx
from gnn import gnn
from datetime import datetime
from scipy.spatial import distance_matrix

def onehot_encoding_node(m, embedding_dim, is_subgraph=True):
    n = m.number_of_nodes()
    H = []
    for i in m.nodes:
        H.append(utils.node_feature(m, i, embedding_dim))
    H = np.array(H)

    # if is_subgraph:
    #     H = np.concatenate([H, np.zeros((n,embedding_dim))], 1)
    # else:
    #     H = np.concatenate([np.zeros((n,embedding_dim)), H], 1)

    return H    

class InferenceGNN():
    def __init__(self, args) -> None:
        if args.ngpu > 0:
            cmd = utils.set_cuda_visible_device(args.ngpu)
            os.environ['CUDA_VISIBLE_DEVICES'] = cmd[:-1]

        self.model = gnn(args)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = utils.initialize_model(self.model, self.device, load_save_file=args.ckpt, gpu=(args.ngpu > 0))

        self.model.eval()
        self.embedding_dim = args.embedding_dim

    def prepare_single_input(self, m1, m2):
        # Prepare subgraph
        n1 = m1.number_of_nodes()
        adj1 = nx.to_numpy_matrix(m1) + np.eye(n1)
        H1 = onehot_encoding_node(m1, self.embedding_dim, is_subgraph=True)

        # Prepare source graph
        n2 = m2.number_of_nodes()
        adj2 = nx.to_numpy_matrix(m2) + np.eye(n2)
        H2 = onehot_encoding_node(m2, self.embedding_dim, is_subgraph=False)
        
        # Aggregation node encoding
        agg_adj1 = np.zeros((n1+n2, n1+n2))
        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2
        agg_adj2 = np.copy(agg_adj1)
        dm = distance_matrix(H1, H2)
        dm_new = np.zeros_like(dm)
        dm_new[dm == 0.0] = 1.0
        agg_adj2[:n1,n1:] = np.copy(dm_new)
        agg_adj2[n1:,:n1] = np.copy(np.transpose(dm_new))
        
        H1 = np.concatenate([H1, np.zeros((n1,self.embedding_dim))], 1)
        H2 = np.concatenate([np.zeros((n2,self.embedding_dim)), H2], 1)
        H = np.concatenate([H1, H2], 0)

        # node indice for aggregation
        valid = np.zeros((n1+n2,))
        valid[:n1] = 1

        sample = {
                  'H':H, \
                  'A1': agg_adj1, \
                  'A2': agg_adj2, \
                  'V': valid, \
                  }

        return sample

    def input_to_tensor(self, batch_input):
        max_natoms = max([len(item['H']) for item in batch_input if item is not None])
        batch_size = len(batch_input)
    
        H = np.zeros((batch_size, max_natoms, batch_input[0]['H'].shape[-1]))
        A1 = np.zeros((batch_size, max_natoms, max_natoms))
        A2 = np.zeros((batch_size, max_natoms, max_natoms))
        V = np.zeros((batch_size, max_natoms))
        
        for i in range(batch_size):
            natom = len(batch_input[i]['H'])
            
            H[i,:natom] = batch_input[i]['H']
            A1[i,:natom,:natom] = batch_input[i]['A1']
            A2[i,:natom,:natom] = batch_input[i]['A2']
            V[i,:natom] = batch_input[i]['V']

        H = torch.from_numpy(H).float()
        A1 = torch.from_numpy(A1).float()
        A2 = torch.from_numpy(A2).float()
        V = torch.from_numpy(V).float()

        H, A1, A2, V = H.to(self.device), A1.to(self.device), A2.to(self.device),V.to(self.device)

        return H, A1, A2, V

    def prepare_multi_input(self, list_subgraphs, list_graphs):
        list_inputs = []
        for li, re in zip(list_subgraphs, list_graphs):
            list_inputs.append(self.prepare_single_input(li, re))

        return list_inputs

    def predict_label(self, list_subgraphs, list_graphs):
        list_inputs = self.prepare_multi_input(list_subgraphs, list_graphs)
        input_tensors = self.input_to_tensor(list_inputs)
        results = self.model.test_model(input_tensors)
        return results.cpu().detach().numpy()

    def predict_embedding(self, list_subgraphs, list_graphs):
        list_inputs = self.prepare_multi_input(list_subgraphs, list_graphs)
        input_tensors = self.input_to_tensor(list_inputs)
        results = self.model.get_refined_adjs2(input_tensors)
        return results.cpu().detach().numpy()
        # from scipy.spatial import distance_matrix
        # results = results.cpu().detach().numpy()
        # return [distance_matrix(results[0], results[0])]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "-c", help="checkpoint for gnn", type=str, default="model/best_large_30_20.pt")
    parser.add_argument("--confidence", help="isomorphism threshold", type=float, default = 0.5)
    parser.add_argument("--mapping_threshold", help="mapping threshold", type=float, default = 1e-30)
    parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
    parser.add_argument("--batch_size", help="batch_size", type=int, default = 32)
    parser.add_argument("--embedding_dim", help="node embedding dim aka number of distinct node label", type=int, default = 20)
    parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 4)
    parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
    parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
    parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
    parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 4.0)
    parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 1.0)
    parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.0)

    args = parser.parse_args()
    print(args)

    inference_gnn = InferenceGNN(args)

    # Load subgraph
    subgraphs = utils.read_graphs("datasets/small/patterns.lg")
    subgraph = subgraphs[1]
    print("subgraph", subgraph != None)
    
    # Load graph
    graphs = utils.read_graphs("datasets/small/graphs.lg")
    graph = graphs[0]
    print("graph", graph != None)
    
    results = inference_gnn.predict_label([subgraph], [graph])
    print("result", results[0] > args.confidence)

    if results[0] > args.confidence:
    # if True:
        interactions = inference_gnn.predict_embedding([subgraph], [graph])
        print("interactions", interactions[0])
        n_subgraph_atom = subgraph.number_of_nodes()
        x_coord, y_coord = np.where(interactions[0] > args.mapping_threshold)

        print("Embedding: (subgraph node, graph node)\n")
        interaction_dict = {}
        for x, y in zip(x_coord, y_coord):
            if x < n_subgraph_atom and y >= n_subgraph_atom:
                interaction_dict[(x, y-n_subgraph_atom)] = interactions[0][x][y]
                # print("(", x, y-n_ligand_atom, ")")

            if x >= n_subgraph_atom and y < n_subgraph_atom and (y, x-n_subgraph_atom) not in interaction_dict:
                interaction_dict[(y, x-n_subgraph_atom)] = interactions[0][x][y]
                # print("(", y, x-n_ligand_atom, ")")

        with open("results/mapping_%s.csv" % datetime.now().strftime("%Y-%m-%dT%H-%M-%S"), "w", encoding="utf8") as f:
            f.write("subgraph_node,graph_node,score\n")
            for key, value in interaction_dict.items():
                f.write("{:d},{:d},{:.3e}\n".format(key[0], key[1], value))

