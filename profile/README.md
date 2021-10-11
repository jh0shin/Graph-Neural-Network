# Profile

## Profiling
### command-line argument
- platform : pyg (Pytorch Geometric) / dgl (Deep Graph Library)
- dataset : ogbn-arxiv (small) ~~/ ogbn-products, ogbn-proteins, ogbn-mag (medium) / ogbn-papers100M (large)~~
- hidden_channel : number of hidden channels (default=128)

```
# PyG ogbn-arxiv profile
python3 arxiv/baseline.py --platform pyg --dataset ogbn-arxiv
# DGL ogbn-arxiv profile
python3 arxiv/baseline.py --platform dgl --dataset ogbn-arxiv
```

> Legacy codes
> ```
> # PyG ogbn-arxiv profile
> python3 arxiv/profile.py --platform pyg --dataset ogbn-arxiv
> # PyG ogbn-products profile
> python3 products/profile.py --platform pyg --dataset ogbn-products
> 
> # DGL ogbn-arxiv profile
> python3 arxiv/profile.py --platform dgl --dataset ogbn-arxiv
> # DGL ogbn-products profile
> python3 products/profile.py --platform dgl --dataset ogbn-products
> ```

## Result
### obgn-arxiv (100 epoches)
```
$ python3 arxiv/baseline.py --platform pyg --dataset ogbn-arxiv
Using backend: pytorch
Accuracy: 0.6715
--- Timer summary -----------------------------------------------
  Event                          |  Count | Average time |  Frac.
- backward                       |    100 |     0.02688s |  30.3%
- forward                        |    100 |     0.05999s |  66.8%
- loss                           |    100 |     0.00193s |   2.2%
- optimizer.step                 |    100 |     0.00096s |   1.1%
-----------------------------------------------------------------
```
```
$ python3 arxiv/baseline.py --platform dgl --dataset ogbn-arxiv
Using backend: pytorch
Accuracy: 0.5909
--- Timer summary -----------------------------------------------
  Event                          |  Count | Average time |  Frac.
- backward                       |    100 |     0.01429s |  36.0%
- forward                        |    100 |     0.02289s |  57.1%
- loss                           |    100 |     0.00191s |   4.8%
- optimizer.step                 |    100 |     0.00085s |   2.1%
-----------------------------------------------------------------
```
