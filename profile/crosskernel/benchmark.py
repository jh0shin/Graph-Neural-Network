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
# MAT_SIZE = 50
# DENSITY = 0.2
# RANDSEED = 34
# ITER = 1
# [MAT_SIZE, DENSITY, RANDSEED, ITER]
settings = [
    # # Matrix size
    # [5000, 0.001, 34, 100],
    # [10000, 0.001, 34, 100],
    # [50000, 0.001, 34, 100],
    # [100000, 0.001, 34, 100],
    # [200000, 0.001, 34, 100],

    # Density
    [100000, 0.0001, 34, 100],
    [100000, 0.0005, 34, 100],
    [100000, 0.001, 34, 100],
    [100000, 0.005, 34, 100],
    [100000, 0.01, 34, 100],
    [100000, 0.05, 34, 100],

    # # Iteration
    # [1000, 0.1, 34, 10],
    # [1000, 0.1, 34, 50],
    # [1000, 0.1, 34, 100],
    # [1000, 0.1, 34, 500],
    # [1000, 0.1, 34, 1000],
    # [1000, 0.1, 34, 5000],
]

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
for MAT_SIZE, DENSITY, RANDSEED, ITER in settings:
    ###########################################
    #         Random matrix generation
    ###########################################
    np.random.seed(RANDSEED)
    torch.random.manual_seed(RANDSEED)

    ### Poor performance for scipy.sparse.random with extremely large shapes
    ### https://github.com/scipy/scipy/issues/9699#issuecomment-763086977
    # spmat = sparse.random(MAT_SIZE, MAT_SIZE, density=DENSITY, dtype=np.float32, random_state=np.random.default_rng(), data_rvs=np.ones)
    spmat = sparse.random(MAT_SIZE, MAT_SIZE, density=DENSITY, dtype=np.float32, random_state=np.random.default_rng())
    mat = torch.rand((MAT_SIZE, 128)).to('cuda')
    
    # print(spmat)
    # print(mat)

    # timer
    timer = Timer(verbosity_level=2, log_fn=log_metric, skip_first=False)

    # # dense * dense
    # _densemat = spmat.todense()

    ###########################################
    #             Kernel setting
    ###########################################
    # pyg kernel
    _spmat = SparseTensor.from_scipy(spmat).to('cuda')

    # dgl kernel
    # op = getattr(ops, 'copy_u_sum')
    op = getattr(ops, 'u_mul_e_sum')
    # _g = dgl.from_scipy(spmat, eweight_name='w').to('cuda')
    _coo = spmat.tocoo()
    row, col, data = _coo.row, _coo.col, _coo.data
    _g = dgl.from_scipy(sparse.coo_matrix((data, (col, row))), eweight_name='w').to('cuda')

    # dgl2 kernel
    op2 = getattr(ops, 'copy_u_sum')
    _g2 = dgl.from_scipy(spmat).to('cuda')

    ###########################################
    #             Iteration
    ###########################################
    for _ in range(ITER):
        # with timer("Dense mm"):
        #     dense_out = torch.from_numpy(_densemat).to('cuda') @ mat

        with timer("PyG spmm"):
            pyg_out = ts.matmul(_spmat, mat, 'add')

        with timer("DGL spmm (u_mul_e_sum)"):
            # dgl_out = op(_g, mat)
            dgl_out = op(_g, mat, _g.edata['w'])

        with timer("DGL spmm 2 (copy_u_sum)"):
            dgl_out2 = op2(_g2, mat)


    '''
    # test
    _spmat2 = SparseTensor.from_torch_sparse_coo_tensor(_g.adj(ctx='cuda')).to('cuda')
    with timer("PyG spmm 2"):
        for _ in range(ITER): pyg_out_2 = ts.matmul(_spmat2, mat, 'add')

    # test2
    # print(type(_g.adj(scipy_fmt='coo')))
    _spmat3 = SparseTensor.from_scipy(_g.adj(scipy_fmt='coo', ctx='cuda')).float().to('cuda')
    with timer("PyG spmm 3"):
        for _ in range(ITER): pyg_out_3 = ts.matmul(_spmat3, mat, 'add')
    '''

    # print(timer)
    print(timer.summary())

    '''
    print('# dense matrix')
    print(mat)

    print('# dense multiplication out')
    print(dense_out)

    print('# pyg spmm')
    # print(_spmat)
    print(pyg_out)

    print('# dgl spmm')
    # print(_g.adj())
    # print(_g.edata['w'])
    print(dgl_out)

    print('# pyg spmm 2')
    # print(_spmat2)
    print(pyg_out_2)

    print('# pyg spmm 3')
    # print(_spmat3)
    print(pyg_out_3)
    '''