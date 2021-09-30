# Profile

## Profiling
### command-line argument
- platform : pyg (Pytorch Geometric) / dgl (Deep Graph Library)
- dataset : ogbn-arxiv (small) / ogbn-products, ogbn-proteins, ogbn-mag (medium) / ogbn-papers100M (large)
- hidden_channel : number of hidden channels (default=128)
```
# PyG ogbn-arxiv profile
python3 profile.py --platform pyg --dataset ogbn-arxiv

# DGL ogbn-arxiv profile
python3 profile.py --platform dgl --dataset ogbn-arxiv
```
