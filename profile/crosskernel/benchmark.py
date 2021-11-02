### Benchmark for each kernel of pyg and dgl

### import
# pyg kernel
import torch
import torch_sparse as ts
from torch_sparse import SparseTensor

# dgl kernel
import dgl
from dgl import ops

# random sparse matrix
import scipy.sparse as sparse
import scipy.stats as stats
import numpy as np

# timer
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from util.timer import Timer

### Params
MAT_SIZE = 5
DENSITY = 0.2
RANDSEED = 34

### Util
def log_metric(name, values, tags={}):
    """Log timeseries data
       This function will be overwritten when called through run.py"""
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{key}:{value:7.3f}")
    values = ", ".join(value_list)
    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)

### Benchmark
# random matrix generation
np.random.seed(RANDSEED)
torch.random.manual_seed(RANDSEED)
spmat = sparse.random(MAT_SIZE, MAT_SIZE, density=DENSITY, dtype=np.float32, data_rvs=np.ones)
mat = torch.rand((MAT_SIZE, MAT_SIZE)).to('cuda')
# print(spmat)
# print(mat)

# timer
timer = Timer(verbosity_level=2, log_fn=log_metric, skip_first=False)

# pyg kernel
_spmat = SparseTensor.from_scipy(spmat).to('cuda')
with timer("PyG spmm"):
    pyg_out = ts.matmul(_spmat, mat, 'add')

# dgl kernel
op = getattr(ops, 'copy_u_sum')
_g = dgl.from_scipy(spmat).to('cuda')
with timer("DGL spmm"):
    dgl_out = op(_g, mat)

# test
_spmat2 = SparseTensor.from_torch_sparse_coo_tensor(_g.adj(ctx='cuda')).to('cuda')
with timer("PyG spmm 2"):
    pyg_out_2 = ts.matmul(_spmat2, mat, 'add')

# test2
print(type(_g.adj(scipy_fmt='coo')))
_spmat3 = SparseTensor.from_scipy(_g.adj(scipy_fmt='coo', ctx='cuda')).float().to('cuda')
with timer("PyG spmm 3"):
    pyg_out_3 = ts.matmul(_spmat3, mat, 'add')


print(timer.summary())

print(_spmat)
print(_spmat2)
print(_spmat3)

print(pyg_out)
print(dgl_out)
print(pyg_out_2)
print(pyg_out_3)