# Profile

## Profiling
### command-line argument
- platform : pyg (Pytorch Geometric) / dgl (Deep Graph Library)
- dataset : ogbn-arxiv (small) / ogbn-products, ogbn-proteins, ogbn-mag (medium) / ogbn-papers100M (large)
- hidden_channel : number of hidden channels (default=128)
```
# PyG ogbn-arxiv profile
python3 arxiv/profile.py --platform pyg --dataset ogbn-arxiv
# PyG ogbn-products profile
python3 products/profile.py --platform pyg --dataset ogbn-products

# DGL ogbn-arxiv profile
python3 arxiv/profile.py --platform dgl --dataset ogbn-arxiv
# DGL ogbn-products profile
python3 products/profile.py --platform dgl --dataset ogbn-products
```

## Result
### obgn-arxiv (100 epoches)
```
$ python3 arxiv/profile.py --platform dgl --dataset ogbn-arxiv
Using backend: pytorch
--- Timer summary -----------------------------------------------
  Event                          |  Count | Average time |  Frac.
- backward                       |    100 |     0.01441s |  35.6%
- forward                        |    100 |     0.02327s |  57.1%
- loss                           |    100 |     0.00194s |   4.8%
- optimizer.step                 |    100 |     0.00088s |   2.2%
-----------------------------------------------------------------
```
```
$ python3 arxiv/profile.py --platform pyg --dataset ogbn-arxiv
Using backend: pytorch
--- Timer summary -----------------------------------------------
  Event                          |  Count | Average time |  Frac.
- backward                       |    100 |     0.02546s |  29.8%
- forward                        |    100 |     0.05805s |  67.2%
- loss                           |    100 |     0.00194s |   2.3%
- optimizer.step                 |    100 |     0.00085s |   1.0%
-----------------------------------------------------------------
```

### obgn-products (100 epoches, 16 sub-clusters)
sub-clustering이나 batch를 사용하지 않을 경우 CUDA out of memory가 발생하였습니다.

```
$ python3 products/profile.py --platform pyg --dataset ogbn-products
--- Timer summary -----------------------------------------------
  Event                          |  Count | Average time |  Frac.
- backward                       |   1600 |     0.02130s |   6.6%
- batch generate                 |    100 |     2.87562s |  55.4%
- forward                        |   1600 |     0.02434s |   7.5%
- loss                           |   1600 |     0.00105s |   0.3%
- optimizer.step                 |   1600 |     0.00093s |   0.3%
- train preprocess               |    100 |     0.00806s |   0.2%
-----------------------------------------------------------------
```
