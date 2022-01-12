# MARGIN

## Maximal Frequent Subgraph Mining

Dear friends,\
This is my optimized implementation of MARGIN.

## How to use

### Install required packages

```
conda env create -f environment.yml
```

### Execute algorithm

```
conda activate margin
python run.py output_file_name memory_log_file_name
                [-h] [--dataset DATASET] [--ckpt CKPT] [--support SUPPORT] [--iso_alg ISO_ALG] [--outdir OUTDIR]
                [--randwalk] [--sortrep] [--confidence CONFIDENCE] [--ngpu NGPU] [--batch_size BATCH_SIZE]
                [--embedding_dim EMBEDDING_DIM] [--n_graph_layer N_GRAPH_LAYER] [--d_graph_layer D_GRAPH_LAYER]
                [--n_FC_layer N_FC_LAYER] [--d_FC_layer D_FC_LAYER] [--initial_mu INITIAL_MU]
                [--initial_dev INITIAL_DEV] [--dropout_rate DROPOUT_RATE]
```

### Get results and memory usage

```
cat output_file_name
python trace.py memory_log_file_name
```
