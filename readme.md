# MARGIN

## Maximal Frequent Subgraph Mining

Dear friends,\
This is my optimized implementation of MARGIN.\

## How to use

### Install required packages

```
pip install -r requirements.txt
```

### Execute algorithm

```
python run.py output_file_name memory_log_file_name
              [-h] [-s MIN_SUPPORT] [-e MIN_EDGE]
              database_file_name
```

### Get results and memory usage

```
cat output_file_name
python trace.py memory_log_file_name
```
