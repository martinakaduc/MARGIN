import time
import torch
import os.path
import numpy as np
import torch.nn as nn
import networkx as nx
from scipy import sparse

def write_graphs(graphs, out_file_name):
    with open(out_file_name, "w", encoding="utf-8") as f:
        for i, g in enumerate(graphs):
            f.write("t # %d\n" % i)
            node_mapping = {}
            for nid, nod in enumerate(g.nodes):
                f.write("v %d %d\n" % (nid, g.nodes[nod]["label"]))
                node_mapping[nod] = nid

            for nod1, nod2 in g.edges:
                nid1 = node_mapping[nod1]
                nid2 = node_mapping[nod2]
                f.write("e %d %d %d\n" % (nid1, nid2, g.edges[(nod1, nod2)]["label"]))

def read_graphs(database_file_name):
    graphs = dict()
    max_size = 0
    with open(database_file_name, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
        tgraph, graph_cnt = None, 0
        graph_size = 0
        for i, line in enumerate(lines):
            cols = line.split(' ')
            if cols[0] == 't':
                if tgraph is not None:
                    graphs[graph_cnt] = tgraph
                    if max_size < graph_size:
                        max_size = graph_size
                    graph_size = 0
                    tgraph = None
                if cols[-1] == '-1':
                    break

                tgraph = nx.Graph()
                graph_cnt = int(cols[2])

            elif cols[0] == 'v':
                tgraph.add_node(int(cols[1]), label=int(cols[2]))
                graph_size += 1

            elif cols[0] == 'e':
                tgraph.add_edge(int(cols[1]), int(cols[2]), label=int(cols[3]))

        # adapt to input files that do not end with 't # -1'
        if tgraph is not None:
            graphs[graph_cnt] = tgraph
            if max_size < graph_size:
                max_size = graph_size

    return graphs

def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    empty = []
    for i in range(8):
        command = 'nvidia-smi -i '+str(i)+' | grep "No running" | wc -l'
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        #print('nvidia-smi -i '+str(i)+' | grep "No running" | wc -l > empty_gpu_check')
        if int(output)==1:
            empty.append(i)
    if len(empty)<ngpus:
        print ('avaliable gpus are less than required')
        exit(-1)
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    return cmd

def initialize_model(model, device, load_save_file=False, gpu=True):
    if load_save_file:
        if not gpu:
            model.load_state_dict(torch.load(load_save_file, map_location=torch.device('cpu'))) 
        else:
            model.load_state_dict(torch.load(load_save_file))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                #nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)

    model.to(device)
    return model

def onehot_encoding(x, max_x):
    onehot_vector = [0] * max_x
    if x < max_x:
        onehot_vector[x] = 1
    return onehot_vector

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def node_feature(m, node_i, max_nodes):
    node = m.nodes[node_i]
    return onehot_encoding(node["label"], max_nodes)

    
