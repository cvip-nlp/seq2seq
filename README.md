# Dataset
```wget https://www.statmt.org/wmt10/training-giga-fren.tar```


# command
```python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=9095 --node_rank 0 train.py -c [path]```