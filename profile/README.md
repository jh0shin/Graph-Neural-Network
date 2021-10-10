# Profile

## Profiling
### command-line argument
- platform : pyg (Pytorch Geometric) / dgl (Deep Graph Library)
- dataset : ogbn-arxiv (small) / ogbn-products, ogbn-proteins, ogbn-mag (medium) / ogbn-papers100M (large)
- hidden_channel : number of hidden channels (default=128)

```
# PyG ogbn-arxiv profile
python3 arxiv/test.py --platform pyg --dataset ogbn-arxiv
# DGL ogbn-arxiv profile
python3 arxiv/test.py --platform dgl --dataset ogbn-arxiv
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
$ python3 arxiv/profile.py --platform pyg --dataset ogbn-arxiv
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
$ python3 arxiv/profile.py --platform dgl --dataset ogbn-arxiv
Using backend: pytorch
/home/junho/.local/lib/python3.8/site-packages/dgl/nn/pytorch/conv/graphconv.py:453: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  rst = self._activation(rst)
Accuracy: 0.5881
--- Timer summary -----------------------------------------------
  Event                          |  Count | Average time |  Frac.
- backward                       |    100 |     0.01450s |  36.1%
- forward                        |    100 |     0.02310s |  56.9%
- loss                           |    100 |     0.00189s |   4.7%
- optimizer.step                 |    100 |     0.00091s |   2.3%
-----------------------------------------------------------------
```
