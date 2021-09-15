# cProfile profile result

## Pytorch geometric

```
PyG profiling result
         57315 function calls (57012 primitive calls) in 4.043 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      300    1.006    0.003    1.254    0.004 loop.py:97(add_remaining_self_loops)
      100    0.789    0.008    0.789    0.008 {method 'run_backward' of 'torch._C._EngineBase' objects}
      300    0.249    0.001    1.744    0.006 gcn_conv.py:30(gcn_norm)
     1200    0.181    0.000    0.181    0.000 {method 'mul_' of 'torch._C._TensorBase' objects}
        1    0.178    0.178    4.043    4.043 pyg_dgl.py:100(pyg_train)
     1200    0.171    0.000    0.171    0.000 {method 'add_' of 'torch._C._TensorBase' objects}
      300    0.133    0.000    2.123    0.007 gcn_conv.py:152(forward)
      100    0.110    0.001    0.725    0.007 _functional.py:53(adam)
      600    0.100    0.000    0.100    0.000 {built-in method cat}
      600    0.094    0.000    0.094    0.000 {method 'scatter_add_' of 'torch._C._TensorBase' objects}
      800    0.093    0.000    0.093    0.000 {built-in method zeros}
      600    0.088    0.000    0.088    0.000 {method 'addcmul_' of 'torch._C._TensorBase' objects}
      600    0.088    0.000    0.088    0.000 {method 'sqrt' of 'torch._C._TensorBase' objects}
      600    0.087    0.000    0.087    0.000 {method 'addcdiv_' of 'torch._C._TensorBase' objects}
      594    0.083    0.000    0.083    0.000 {method 'zero_' of 'torch._C._TensorBase' objects}
      300    0.052    0.000    0.055    0.000 gcn_conv.py:190(message)
      300    0.052    0.000    0.052    0.000 {method 'repeat' of 'torch._C._TensorBase' objects}
      300    0.050    0.000    0.050    0.000 {method 'index_select' of 'torch._C._TensorBase' objects}
      300    0.048    0.000    0.048    0.000 {built-in method ones}
      300    0.048    0.000    0.048    0.000 {method 'pow_' of 'torch._C._TensorBase' objects}
      300    0.047    0.000    0.047    0.000 {built-in method arange}
      ...
```

[gcn_conv.py:30(gcn_norm)](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gcn_conv.py#L30)

-> [mul (imported from torch_sparse.matmul as mul)](https://github.com/rusty1s/pytorch_sparse/blob/master/torch_sparse/matmul.py#L123)

-> @torch.jit._overload를 통한 torch._C 호출

[gcn_conv.py:152(forward)](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gcn_conv.py#L154)

-> [gcn_norm (in same file)](https://github.com/pyg-team/pytorch_geometric/blob/c4a2aae63351c6996f2e4b38580981f293918fcb/torch_geometric/nn/conv/gcn_conv.py#L17)

-> @torch.jit._overload를 통한 torch._C 호출

@torch.jit._overload

-> [pytorch_sparse/csrc/cuda/spmm_cuda.cu](https://github.com/rusty1s/pytorch_sparse/blob/master/csrc/cuda/spmm_cuda.cu#L14)

## Deep Graph Library

```
DGL profiling result
         242234 function calls (241931 primitive calls) in 4.094 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      100    0.872    0.009    0.872    0.009 {method 'run_backward' of 'torch._C._EngineBase' objects}
      300    0.356    0.001    2.060    0.007 graphconv.py:337(forward)
      600    0.209    0.000    0.209    0.000 {method 'item' of 'torch._C._TensorBase' objects}
        1    0.186    0.186    4.094    4.094 pyg_dgl.py:89(dgl_train)
     1200    0.181    0.000    0.181    0.000 {method 'add_' of 'torch._C._TensorBase' objects}
     1200    0.180    0.000    0.180    0.000 {method 'mul_' of 'torch._C._TensorBase' objects}
      900    0.143    0.000    0.143    0.000 {built-in method arange}
      100    0.114    0.001    0.743    0.007 _functional.py:53(adam)
      600    0.101    0.000    0.147    0.000 heterograph_index.py:560(in_degrees)
      600    0.100    0.000    0.100    0.000 {built-in method pow}
      600    0.098    0.000    0.098    0.000 {method 'clamp' of 'torch._C._TensorBase' objects}
      600    0.096    0.000    0.096    0.000 {method 'float' of 'torch._C._TensorBase' objects}
      300    0.091    0.000    0.091    0.000 {built-in method sum}
      600    0.091    0.000    0.091    0.000 {method 'addcdiv_' of 'torch._C._TensorBase' objects}
      600    0.088    0.000    0.088    0.000 {method 'addcmul_' of 'torch._C._TensorBase' objects}
      600    0.088    0.000    0.088    0.000 {method 'sqrt' of 'torch._C._TensorBase' objects}
      594    0.082    0.000    0.082    0.000 {method 'zero_' of 'torch._C._TensorBase' objects}
      300    0.076    0.000    0.076    0.000 {built-in method matmul}
     7200    0.063    0.000    0.066    0.000 ndarray.py:177(<genexpr>)
      300    0.060    0.000    0.124    0.000 sparse.py:77(_gspmm)
      500    0.054    0.000    0.054    0.000 {built-in method zeros}
      300    0.053    0.000    0.053    0.000 {built-in method min}
      ...
```

[graphconv.py:337(forward)](https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/graphconv.py#L337)

-> forward - matmul defined in [backend/pytorch/__init__.py](https://github.com/dmlc/dgl/blob/ad61a9a55092849111e4e7277942879fb82c9ed7/tests/backend/pytorch/__init__.py#L66)

[dgl/sparse.py:77(_gspmm)](https://github.com/dmlc/dgl/blob/master/python/dgl/sparse.py#L88)

-> _CAPI_DGLKernelSpMM

-> ...

-> [/src/array/cuda/spmm.cu](https://github.com/dmlc/dgl/blob/master/src/array/cuda/spmm.cu)

-> CusparseCsrmm2 (?)
