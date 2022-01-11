import os
import utils
import torch
import numpy as np
import networkx as nx
from gnn import gnn
from scipy.spatial import distance_matrix

def onehot_encoding_node(m, embedding_dim, is_subgraph=True):
    n = m.number_of_nodes()
    H = []
    for i in m.nodes:
        H.append(utils.node_feature(m, i, embedding_dim))
    H = np.array(H)

    if is_subgraph:
        H = np.concatenate([H, np.zeros((n,embedding_dim))], 1)
    else:
        H = np.concatenate([np.zeros((n,embedding_dim)), H], 1)

    return H    

class InferenceGNN():
    def __init__(self, args) -> None:
        if args.ngpu > 0:
            cmd = utils.set_cuda_visible_device(args.ngpu)
            os.environ['CUDA_VISIBLE_DEVICES'] = cmd[:-1]

        self.model = gnn(args)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = utils.initialize_model(self.model, device, load_save_file=args.ckpt, gpu=(args.ngpu > 0))

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
        H = np.concatenate([H1, H2], 0)
        agg_adj1 = np.zeros((n1+n2, n1+n2))
        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2
        agg_adj2 = np.copy(agg_adj1)
        dm = distance_matrix(H1, H2)
        agg_adj2[:n1,n1:] = np.copy(dm)
        agg_adj2[n1:,:n1] = np.copy(np.transpose(dm))

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

# if __name__ == '__main__':
#     args = gnn_parser.parse_args()
#     print(args)

#     inference_gnn = InferenceGNN(args)

#     # Load subgraph
#     subgraphs_sdf = SDMolSupplier("dude/egfr_CHEMBL144760/CHEMBL144760.sdf" )
#     subgraph = subgraphs_sdf[0]
#     print("subgraph", subgraph != None)
    
#     # Load graph
#     graph = MolFromPDBFile("dude/egfr_CHEMBL144760/egfr.pdb")
#     print("graph", graph != None)
    
#     results = inference_gnn.predict_label([subgraph], [graph])
#     print("result", results[0] > args.active_threshold)

#     if results[0] > args.active_threshold:
#         interactions = inference_gnn.predict_embedding([subgraph], [graph])
#         # print("interactions", interactions[0])
#         n_subgraph_atom = subgraph.GetNumAtoms()
#         x_coord, y_coord = np.where(interactions[0] > args.interaction_threshold)

#         print("Embedding: (subgraph node, graph node)\n")
#         for x, y in zip(x_coord, y_coord):
#             if x < n_subgraph_atom and y >= n_subgraph_atom:
#                 print("(", x, y-n_subgraph_atom, ")")

#             if x >= n_subgraph_atom and y < n_subgraph_atom:
#                 print("(", y, x-n_subgraph_atom, ")")
