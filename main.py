import os
import time
import argparse
from margin_vf2 import MARGIN_VF2
from margin_gnn import MARGIN_GNN
from utils import read_graphs, write_graphs

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", help="input dataset", type=str, default="datasets/base/graphs.lg")
parser.add_argument("--ckpt", "-c", help="checkpoint for gnn", type=str, default="model/best_large.pt")
parser.add_argument("--support", "-s", help="support threshold", type=float, default=0.3)
parser.add_argument("--iso_alg", "-i", help="isomorphsim algorithm", type=str, default="vf2")
parser.add_argument("--outdir", "-o", help="output dir", type=str, default="results")
parser.add_argument("--randwalk", "-r", help="Enable random walk", action="store_true")
parser.add_argument("--sortrep", "-sr", help="Enable representative sorting", action="store_true")

parser.add_argument("--confidence", help="isomorphism threshold", type=int, default = 0.5)
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

def main(args):
    start = time.time()
    data = read_graphs(args.dataset)
    data_read_time = time.time()

    if args.iso_alg == "vf2":
        margin = MARGIN_VF2(support=args.support, randwalk=args.randwalk, sortrep=args.sortrep)
    elif args.iso_alg == "gnn":
        margin = MARGIN_GNN(support=args.support, gnn_args=args, 
                            randwalk=args.randwalk, sortrep=args.sortrep)
    else:
        raise AssertionError("Undefined algorithm %s" % args.iso_alg)

    results = margin.run(data)
    margin_time = time.time()

    output_file = os.path.join(args.outdir, args.dataset.split("/")[-2] + ".lg")
    write_graphs(results, output_file)

    print("Data Reading Time:", data_read_time-start)
    print("Margin Time:", margin_time-data_read_time)

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    main(args)