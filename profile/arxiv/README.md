# ogbn-arxiv

### command-line argument
- platform : pyg (Pytorch Geometric) / dgl (Deep Graph Library)
- dataset : ogbn-arxiv (small) ~~/ ogbn-products, ogbn-proteins, ogbn-mag (medium) / ogbn-papers100M (large)~~
- hidden_channel : number of hidden channels (default=128)

```
# PyG ogbn-arxiv profile
python3 arxiv/baseline_{profiler}.py --platform pyg --dataset ogbn-arxiv
# DGL ogbn-arxiv profile
python3 arxiv/baseline_{profiler}.py --platform dgl --dataset ogbn-arxiv
```


## Result
### baseline (custom timer class) (100 epoches)
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

### cprofile (100 epoches)
```
$ python3 arxiv/baseline_cprofile.py --platform pyg --dataset ogbn-arxiv
Using backend: pytorch
Accuracy: 0.6670
         153697 function calls (153090 primitive calls) in 8.744 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     6000    4.556    0.001    4.556    0.001 {method 'item' of 'torch._C._TensorBase' objects}
      300    1.894    0.006    1.894    0.006 {method 'argsort' of 'torch._C._TensorBase' objects}
      100    0.787    0.008    0.787    0.008 {method 'run_backward' of 'torch._C._EngineBase' objects}
      300    0.485    0.002    1.246    0.004 diag.py:35(set_diag)
      300    0.286    0.001    0.513    0.002 diag.py:9(remove_diag)
     6000    0.173    0.000    0.173    0.000 {method 'max' of 'torch._C._TensorBase' objects}
      300    0.085    0.000    0.085    0.000 {built-in method torch._ops.torch_scatter.gather_csr}
      300    0.039    0.000    7.827    0.026 gcn_conv.py:152(forward)
     3000    0.039    0.000    4.779    0.002 storage.py:33(__init__)
     1800    0.027    0.000    0.027    0.000 {method 'mul_' of 'torch._C._TensorBase' objects}
     1500    0.021    0.000    0.021    0.000 {method 'add_' of 'torch._C._TensorBase' objects}
      300    0.018    0.000    0.018    0.000 {built-in method torch._ops.torch_sparse.spmm_sum}
      300    0.018    0.000    1.912    0.006 storage.py:347(csr2csc)
      300    0.017    0.000    5.817    0.019 gcn_conv.py:30(gcn_norm)
      600    0.017    0.000    0.017    0.000 {built-in method torch._ops.torch_sparse.ind2ptr}
      300    0.015    0.000    0.023    0.000 storage.py:312(colptr)
      300    0.015    0.000    0.015    0.000 {built-in method torch._ops.torch_sparse.non_diag_mask}
      300    0.015    0.000    0.015    0.000 {built-in method torch._ops.torch_scatter.segment_sum_csr}
      100    0.013    0.000    0.063    0.001 _functional.py:53(adam)
      600    0.012    0.000    0.573    0.001 mul.py:8(mul)
      300    0.010    0.000    0.010    0.000 {method 'new_full' of 'torch._C._TensorBase' objects}
      300    0.010    0.000    0.010    0.000 {method 'pow_' of 'torch._C._TensorBase' objects}
      300    0.008    0.000    0.008    0.000 {built-in method full}
      300    0.008    0.000    0.008    0.000 {method 'masked_fill_' of 'torch._C._TensorBase' objects}
      600    0.007    0.000    0.007    0.000 {method 'sqrt' of 'torch._C._TensorBase' objects}
  700/100    0.007    0.000    7.856    0.079 module.py:866(_call_impl)
      600    0.007    0.000    0.007    0.000 {method 'addcdiv_' of 'torch._C._TensorBase' objects}
      900    0.007    0.000    0.007    0.000 {method 'new_empty' of 'torch._C._TensorBase' objects}
      300    0.007    0.000    0.007    0.000 {built-in method arange}
    11900    0.007    0.000    0.007    0.000 {method 'size' of 'torch._C._TensorBase' objects}
      600    0.007    0.000    0.007    0.000 {method 'addcmul_' of 'torch._C._TensorBase' objects}
      594    0.006    0.000    0.006    0.000 {method 'zero_' of 'torch._C._TensorBase' objects}
      200    0.006    0.000    0.006    0.000 {built-in method relu_}
      900    0.005    0.000    4.062    0.005 storage.py:213(set_value)
      100    0.005    0.000    7.855    0.079 sequential.py:99(forward)
      700    0.004    0.000    0.004    0.000 {method 'squeeze' of 'torch._C._TensorBase' objects}
      100    0.004    0.000    0.004    0.000 {method 'log_softmax' of 'torch._C._TensorBase' objects}
     1500    0.004    0.000    0.485    0.000 tensor.py:27(from_storage)
     1800    0.004    0.000    0.006    0.000 tensor.py:212(sizes)
    11100    0.004    0.000    0.004    0.000 {method 'contiguous' of 'torch._C._TensorBase' objects}
      300    0.004    0.000    1.262    0.004 diag.py:82(fill_diag)
      100    0.003    0.000    0.003    0.000 {built-in method torch._C._nn.nll_loss}
     1500    0.003    0.000    0.481    0.000 tensor.py:15(__init__)
      600    0.003    0.000    0.003    0.000 {method 'view' of 'torch._C._TensorBase' objects}
    15100    0.003    0.000    0.003    0.000 {method 'numel' of 'torch._C._TensorBase' objects}
      300    0.003    0.000    0.003    0.000 message_passing.py:138(__collect__)
     4182    0.003    0.000    0.004    0.000 tensor.py:906(grad)
      300    0.003    0.000    1.969    0.007 message_passing.py:186(propagate)
      300    0.003    0.000    1.956    0.007 matmul.py:7(spmm_sum)
      100    0.003    0.000    0.003    0.000 {built-in method ones_like}
      200    0.003    0.000    0.003    0.000 {built-in method torch._ops.profiler._record_function_enter}
      100    0.002    0.000    0.069    0.001 adam.py:55(step)
      200    0.002    0.000    0.002    0.000 {built-in method zeros}
      600    0.002    0.000    0.003    0.000 inspector.py:52(distribute)
      900    0.002    0.000    4.363    0.005 tensor.py:169(set_value)
      700    0.002    0.000    0.002    0.000 {built-in method torch._C._get_tracing_state}
      100    0.002    0.000    0.078    0.001 optimizer.py:84(wrapper)
     1000    0.002    0.000    0.002    0.000 module.py:934(__getattr__)
      100    0.002    0.000    0.012    0.000 optimizer.py:189(zero_grad)
     7300    0.001    0.000    0.001    0.000 {method 'dim' of 'torch._C._TensorBase' objects}
     8000    0.001    0.000    0.001    0.000 {built-in method builtins.len}
      300    0.001    0.000    3.918    0.013 tensor.py:207(fill_value)
      300    0.001    0.000    0.002    0.000 message_passing.py:86(__check_input__)
      600    0.001    0.000    0.001    0.000 {method 'to' of 'torch._C._TensorBase' objects}
     2400    0.001    0.000    0.001    0.000 tensor.py:173(sparse_sizes)
     1500    0.001    0.000    0.009    0.000 storage.py:179(rowptr)
      100    0.001    0.000    0.792    0.008 tensor.py:195(backward)
      900    0.001    0.000    0.001    0.000 tensor.py:149(csr)
      200    0.001    0.000    0.001    0.000 {built-in method torch._ops.profiler._record_function_exit}
      300    0.001    0.000    0.026    0.000 reduce.py:8(reduction)
      600    0.001    0.000    0.001    0.000 tensor.py:146(coo)
     1200    0.001    0.000    0.001    0.000 tensor.py:176(sparse_size)
      100    0.001    0.000    0.791    0.008 __init__.py:68(backward)
      200    0.001    0.000    0.007    0.000 activation.py:101(forward)
      100    0.001    0.000    0.005    0.000 functional.py:2312(nll_loss)
      600    0.001    0.000    0.001    0.000 {method 'update' of 'dict' objects}
      300    0.001    0.000    1.958    0.007 gcn_conv.py:193(message_and_aggregate)
     1500    0.001    0.000    0.005    0.000 tensor.py:220(size)
      100    0.001    0.000    0.001    0.000 grad_mode.py:114(__init__)
     3516    0.001    0.000    0.001    0.000 {built-in method builtins.isinstance}
      300    0.001    0.000    0.085    0.000 segment_csr.py:112(gather_csr)
      300    0.001    0.000    0.001    0.000 {built-in method builtins.min}
      300    0.001    0.000    0.016    0.000 segment_csr.py:6(segment_sum_csr)
      200    0.001    0.000    0.003    0.000 profiler.py:615(__enter__)
      300    0.001    0.000    1.957    0.007 matmul.py:121(matmul)
      100    0.001    0.000    0.004    0.000 __init__.py:28(_make_grads)
      100    0.001    0.000    0.071    0.001 grad_mode.py:24(decorate_context)
     5188    0.000    0.000    0.000    0.000 {built-in method torch._C._has_torch_function_unary}
      600    0.000    0.000    0.000    0.000 sequential.py:109(<dictcomp>)
     4800    0.000    0.000    0.000    0.000 storage.py:198(value)
     3600    0.000    0.000    0.000    0.000 storage.py:228(sparse_sizes)
      300    0.000    0.000    0.016    0.000 segment_csr.py:35(segment_csr)
      200    0.000    0.000    0.006    0.000 functional.py:1195(relu)
      200    0.000    0.000    0.003    0.000 profiler.py:607(__init__)
      300    0.000    0.000    0.000    0.000 tensor.py:343(device)
      900    0.000    0.000    0.000    0.000 storage.py:11(get_layout)
      300    0.000    0.000    0.027    0.000 reduce.py:70(sum)
      300    0.000    0.000    0.001    0.000 tensor.py:226(nnz)
     1500    0.000    0.000    0.000    0.000 storage.py:163(row)
      594    0.000    0.000    0.000    0.000 {method 'requires_grad_' of 'torch._C._TensorBase' objects}
      100    0.000    0.000    0.005    0.000 activation.py:1270(forward)
      300    0.000    0.000    1.957    0.007 matmul.py:63(spmm)
     4282    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}
      300    0.000    0.000    0.000    0.000 tensor.py:161(has_value)
     2400    0.000    0.000    0.000    0.000 storage.py:192(col)
      600    0.000    0.000    0.000    0.000 sequential.py:106(<listcomp>)
      104    0.000    0.000    0.000    0.000 {method 'format' of 'str' objects}
      608    0.000    0.000    0.000    0.000 {method 'items' of 'collections.OrderedDict' objects}
      200    0.000    0.000    0.001    0.000 profiler.py:619(__exit__)
      200    0.000    0.000    0.000    0.000 grad_mode.py:200(__init__)
     2900    0.000    0.000    0.000    0.000 {method 'values' of 'collections.OrderedDict' objects}
     3100    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      100    0.000    0.000    0.004    0.000 functional.py:1651(log_softmax)
      100    0.000    0.000    0.001    0.000 grad_mode.py:119(__enter__)
      606    0.000    0.000    0.000    0.000 tensor.py:593(__hash__)
       12    0.000    0.000    0.000    0.000 {built-in method zeros_like}
      100    0.000    0.000    0.000    0.000 grad_mode.py:123(__exit__)
      600    0.000    0.000    0.000    0.000 {built-in method math.sqrt}
      300    0.000    0.000    0.000    0.000 storage.py:195(has_value)
      100    0.000    0.000    0.000    0.000 container.py:186(__iter__)
      300    0.000    0.000    0.000    0.000 {built-in method torch._C.is_grad_enabled}
        4    0.000    0.000    0.000    0.000 {built-in method torch._C._jit_get_operation}
      610    0.000    0.000    0.000    0.000 {built-in method builtins.id}
      100    0.000    0.000    0.000    0.000 sequential.py:102(<dictcomp>)
      100    0.000    0.000    0.000    0.000 _reduction.py:7(get_enum)
      624    0.000    0.000    0.000    0.000 {method 'get' of 'dict' objects}
      300    0.000    0.000    0.000    0.000 message_passing.py:300(update)
      200    0.000    0.000    0.000    0.000 {built-in method torch._C._set_grad_enabled}
      100    0.000    0.000    0.000    0.000 __init__.py:60(_tensor_or_tensors_to_tuple)
      100    0.000    0.000    0.000    0.000 _jit_internal.py:833(is_scripting)
        4    0.000    0.000    0.000    0.000 _ops.py:56(__getattr__)
      100    0.000    0.000    0.000    0.000 {built-in method builtins.iter}
        8    0.000    0.000    0.000    0.000 module.py:950(__setattr__)
      100    0.000    0.000    0.000    0.000 {built-in method torch._C._has_torch_function_variadic}
      8/1    0.000    0.000    0.000    0.000 module.py:1432(train)
       15    0.000    0.000    0.000    0.000 module.py:1347(named_children)
        4    0.000    0.000    0.000    0.000 _builtins.py:141(_register_builtin)
       15    0.000    0.000    0.000    0.000 module.py:1338(children)
        1    0.000    0.000    0.000    0.000 _ops.py:52(__init__)
        1    0.000    0.000    0.000    0.000 _ops.py:75(__getattr__)
        5    0.000    0.000    0.000    0.000 {built-in method builtins.setattr}
        4    0.000    0.000    0.000    0.000 _builtins.py:110(_get_builtin_table)
        7    0.000    0.000    0.000    0.000 {method 'add' of 'set' objects}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
```
```
$ python3 arxiv/baseline_cprofile.py --platform dgl --dataset ogbn-arxiv
Using backend: pytorch
Accuracy: 0.5876
         236439 function calls (236134 primitive calls) in 3.799 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      300    2.375    0.008    2.821    0.009 graphconv.py:337(forward)
      100    0.873    0.009    0.873    0.009 {method 'run_backward' of 'torch._C._EngineBase' objects}
      600    0.048    0.000    0.048    0.000 {method 'item' of 'torch._C._TensorBase' objects}
      900    0.027    0.000    0.027    0.000 {built-in method arange}
      300    0.025    0.000    0.025    0.000 {built-in method matmul}
     1200    0.015    0.000    0.015    0.000 {method 'mul_' of 'torch._C._TensorBase' objects}
      600    0.015    0.000    0.015    0.000 {built-in method pow}
      600    0.015    0.000    0.037    0.000 heterograph_index.py:560(in_degrees)
      300    0.015    0.000    0.015    0.000 {built-in method min}
      600    0.014    0.000    0.014    0.000 {method 'float' of 'torch._C._TensorBase' objects}
      300    0.013    0.000    0.031    0.000 sparse.py:77(_gspmm)
      600    0.013    0.000    0.013    0.000 {method 'clamp' of 'torch._C._TensorBase' objects}
     1200    0.013    0.000    0.013    0.000 {method 'add_' of 'torch._C._TensorBase' objects}
      100    0.013    0.000    0.064    0.001 _functional.py:53(adam)
      500    0.012    0.000    0.012    0.000 {built-in method zeros}
      300    0.012    0.000    0.012    0.000 {built-in method sum}
     3000    0.011    0.000    0.011    0.000 tensor.py:94(to_backend_ctx)
     1200    0.010    0.000    0.012    0.000 {built-in method torch._C._from_dlpack}
     2700    0.010    0.000    0.010    0.000 heterograph_index.py:160(dtype)
      300    0.009    0.000    0.009    0.000 {method 'any' of 'torch._C._TensorBase' objects}
      300    0.009    0.000    0.009    0.000 {method 'type' of 'torch._C._TensorBase' objects}
     3600    0.009    0.000    0.021    0.000 ndarray.py:174(shape)
     7200    0.009    0.000    0.009    0.000 __init__.py:506(cast)
      600    0.009    0.000    0.009    0.000 {method 'addcmul_' of 'torch._C._TensorBase' objects}
      300    0.008    0.000    0.022    0.000 heterograph_index.py:315(has_nodes)
      300    0.008    0.000    0.017    0.000 heterograph_index.py:580(out_degrees)
      600    0.007    0.000    0.007    0.000 {method 'sqrt' of 'torch._C._TensorBase' objects}
      600    0.007    0.000    0.007    0.000 {method 'addcdiv_' of 'torch._C._TensorBase' objects}
      594    0.006    0.000    0.006    0.000 {method 'zero_' of 'torch._C._TensorBase' objects}
     3300    0.005    0.000    0.007    0.000 heterograph_index.py:62(metagraph)
      200    0.005    0.000    0.005    0.000 {built-in method relu}
      300    0.004    0.000    0.174    0.001 heterograph.py:3510(out_degrees)
      300    0.004    0.000    0.039    0.000 {built-in method apply}
     1500    0.004    0.000    0.021    0.000 checks.py:9(prepare_tensor)
  400/100    0.004    0.000    2.828    0.028 module.py:866(_call_impl)
     3000    0.004    0.000    0.006    0.000 heterograph_index.py:171(ctx)
      600    0.004    0.000    0.004    0.000 {built-in method reshape}
     1200    0.004    0.000    0.039    0.000 tensor.py:339(zerocopy_from_dgl_ndarray)
      100    0.003    0.000    0.003    0.000 {built-in method torch._C._nn.nll_loss}
     7200    0.003    0.000    0.006    0.000 ndarray.py:177(<genexpr>)
      900    0.003    0.000    0.047    0.000 view.py:40(__call__)
      600    0.003    0.000    0.018    0.000 heterograph.py:4083(_set_n_repr)
    13608    0.003    0.000    0.006    0.000 {built-in method builtins.isinstance}
      600    0.003    0.000    0.081    0.000 heterograph.py:3426(in_degrees)
     4182    0.003    0.000    0.004    0.000 tensor.py:906(grad)
      200    0.003    0.000    0.003    0.000 {built-in method torch._ops.profiler._record_function_enter}
     2700    0.003    0.000    0.011    0.000 heterograph_index.py:77(number_of_etypes)
      100    0.003    0.000    0.003    0.000 {method 'log_softmax' of 'torch._C._TensorBase' objects}
     3000    0.003    0.000    0.020    0.000 heterograph.py:5318(device)
      300    0.002    0.000    0.078    0.000 heterograph.py:2682(has_nodes)
      100    0.002    0.000    0.069    0.001 adam.py:55(step)
    10700    0.002    0.000    0.005    0.000 {built-in method builtins.len}
      100    0.002    0.000    0.014    0.000 optimizer.py:189(zero_grad)
     2400    0.002    0.000    0.012    0.000 heterograph.py:2631(idtype)
      300    0.002    0.000    0.034    0.000 sparse.py:120(forward)
     1800    0.002    0.000    0.002    0.000 heterograph_index.py:285(number_of_nodes)
     1800    0.002    0.000    0.006    0.000 tensor.py:333(zerocopy_to_dgl_ndarray)
     1200    0.002    0.000    0.003    0.000 heterograph.py:1094(to_canonical_etype)
     1800    0.002    0.000    0.010    0.000 heterograph.py:1241(get_etype_id)
     3000    0.002    0.000    0.002    0.000 runtime_ctypes.py:140(__new__)
      100    0.002    0.000    0.002    0.000 {built-in method ones_like}
     3600    0.002    0.000    0.002    0.000 {built-in method builtins.getattr}
     2400    0.002    0.000    0.002    0.000 {built-in method _abc._abc_instancecheck}
      300    0.002    0.000    0.073    0.000 heterograph.py:4753(update_all)
      300    0.002    0.000    0.049    0.000 core.py:248(invoke_gspmm)
      300    0.002    0.000    0.008    0.000 heterograph.py:2138(__getitem__)
     3300    0.002    0.000    0.002    0.000 graph_index.py:36(__new__)
      600    0.002    0.000    0.002    0.000 graph_index.py:327(find_edge)
     2700    0.002    0.000    0.002    0.000 graph_index.py:168(number_of_edges)
      300    0.002    0.000    0.002    0.000 heterograph.py:86(_init)
      600    0.002    0.000    0.003    0.000 frame.py:35(infer_scheme)
     1800    0.002    0.000    0.002    0.000 {dgl._ffi._cy3.core._from_dlpack}
      300    0.001    0.000    0.002    0.000 heterograph_index.py:81(get_relation_graph)
     1800    0.001    0.000    0.001    0.000 {built-in method torch._C._to_dlpack}
      600    0.001    0.000    0.003    0.000 heterograph.py:1849(srcdata)
      600    0.001    0.000    0.005    0.000 frame.py:300(__init__)
      300    0.001    0.000    0.052    0.000 core.py:300(message_passing)
      600    0.001    0.000    0.008    0.000 heterograph.py:5546(local_scope)
      100    0.001    0.000    2.827    0.028 utils.py:199(forward)
     2700    0.001    0.000    0.001    0.000 tensor.py:72(shape)
      600    0.001    0.000    0.006    0.000 frame.py:466(update_column)
      600    0.001    0.000    0.001    0.000 tensor.py:568(__len__)
      400    0.001    0.000    0.001    0.000 {built-in method torch._C._get_tracing_state}
      600    0.001    0.000    0.008    0.000 _collections_abc.py:824(update)
      100    0.001    0.000    0.075    0.001 optimizer.py:84(wrapper)
     1302    0.001    0.000    0.001    0.000 {method 'format' of 'str' objects}
     2400    0.001    0.000    0.001    0.000 tensor.py:81(context)
     2400    0.001    0.000    0.001    0.000 tensor.py:75(dtype)
     3300    0.001    0.000    0.002    0.000 tensor.py:78(ndim)
     2700    0.001    0.000    0.001    0.000 base.py:25(is_all)
     1200    0.001    0.000    0.005    0.000 frame.py:265(create)
     1200    0.001    0.000    0.003    0.000 frame.py:92(__init__)
     4600    0.001    0.000    0.001    0.000 {method 'dim' of 'torch._C._TensorBase' objects}
     4200    0.001    0.000    0.001    0.000 {built-in method __new__ of type object at 0x907780}
      200    0.001    0.000    0.001    0.000 {built-in method torch._ops.profiler._record_function_exit}
     1800    0.001    0.000    0.001    0.000 {method 'contiguous' of 'torch._C._TensorBase' objects}
      900    0.001    0.000    0.001    0.000 module.py:934(__getattr__)
      600    0.001    0.000    0.002    0.000 heterograph.py:1915(dstdata)
     1500    0.001    0.000    0.001    0.000 tensor.py:69(is_tensor)
     1200    0.001    0.000    0.001    0.000 {built-in method builtins.min}
      300    0.001    0.000    0.001    0.000 message.py:102(copy_u)
      300    0.001    0.000    0.002    0.000 contextlib.py:117(__exit__)
      300    0.001    0.000    0.011    0.000 view.py:68(__setitem__)
     1200    0.001    0.000    0.001    0.000 frame.py:98(__len__)
      100    0.001    0.000    0.001    0.000 {method 'squeeze' of 'torch._C._TensorBase' objects}
      300    0.001    0.000    0.003    0.000 heterograph.py:40(__init__)
     2400    0.001    0.000    0.003    0.000 abc.py:96(__instancecheck__)
      300    0.001    0.000    0.016    0.000 tensor.py:148(min)
      200    0.001    0.000    0.004    0.000 profiler.py:615(__enter__)
     1200    0.001    0.000    0.002    0.000 __init__.py:144(_lazy_init)
     1200    0.001    0.000    0.001    0.000 view.py:51(__init__)
      100    0.001    0.000    0.004    0.000 functional.py:2312(nll_loss)
      600    0.001    0.000    0.006    0.000 frame.py:566(clone)
      900    0.001    0.000    0.028    0.000 tensor.py:313(arange)
      300    0.001    0.000    0.001    0.000 contextlib.py:82(__init__)
     1200    0.001    0.000    0.001    0.000 heterograph.py:955(srctypes)
     1200    0.001    0.000    0.001    0.000 __init__.py:107(is_initialized)
      300    0.001    0.000    0.041    0.000 spmm.py:35(gspmm)
      600    0.001    0.000    0.002    0.000 view.py:57(__getitem__)
      100    0.001    0.000    0.876    0.009 __init__.py:68(backward)
     1200    0.001    0.000    0.040    0.000 __init__.py:121(from_dgl_nd)
      100    0.001    0.000    0.877    0.009 tensor.py:195(backward)
     5788    0.001    0.000    0.001    0.000 {built-in method torch._C._has_torch_function_unary}
     4512    0.001    0.000    0.001    0.000 {method 'get' of 'dict' objects}
      100    0.001    0.000    0.003    0.000 __init__.py:28(_make_grads)
     1200    0.001    0.000    0.001    0.000 heterograph.py:1191(get_ntype_id_from_src)
      300    0.001    0.000    0.039    0.000 sparse.py:487(gspmm)
      300    0.001    0.000    0.034    0.000 autocast_mode.py:204(decorate_fwd)
     1500    0.001    0.000    0.001    0.000 heterograph.py:1216(get_ntype_id_from_dst)
      600    0.001    0.000    0.001    0.000 heterograph.py:1720(dstnodes)
      100    0.001    0.000    0.001    0.000 grad_mode.py:114(__init__)
     1800    0.001    0.000    0.002    0.000 ndarray.py:153(from_dlpack)
      300    0.001    0.000    0.002    0.000 view.py:164(__init__)
      600    0.001    0.000    0.002    0.000 frame.py:308(<dictcomp>)
      600    0.001    0.000    0.001    0.000 frame.py:225(clone)
      300    0.001    0.000    0.001    0.000 heterograph.py:1157(get_ntype_id)
     1200    0.001    0.000    0.002    0.000 __init__.py:131(init)
      300    0.000    0.000    0.002    0.000 heterograph.py:2070(edata)
     1200    0.000    0.000    0.005    0.000 __init__.py:118(to_dgl_nd)
     1200    0.000    0.000    0.001    0.000 heterograph.py:1006(dsttypes)
     1200    0.000    0.000    0.000    0.000 {built-in method torch._C._cuda_isInBadFork}
      900    0.000    0.000    0.001    0.000 sparse.py:62(to_dgl_nd_for_write)
      100    0.000    0.000    0.071    0.001 grad_mode.py:24(decorate_context)
      300    0.000    0.000    0.000    0.000 heterograph.py:2131(<listcomp>)
      300    0.000    0.000    0.001    0.000 heterograph.py:2130(_find_etypes)
      600    0.000    0.000    0.048    0.000 tensor.py:48(as_scalar)
      600    0.000    0.000    0.001    0.000 message.py:97(name)
     4282    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}
      300    0.000    0.000    0.001    0.000 internal.py:553(expand_as_pair)
      300    0.000    0.000    0.001    0.000 contextlib.py:238(helper)
      900    0.000    0.000    0.000    0.000 view.py:17(__init__)
      300    0.000    0.000    0.000    0.000 message.py:78(__init__)
     1200    0.000    0.000    0.000    0.000 {method 'to_dlpack' of 'dgl._ffi._cy3.core.NDArrayBase' objects}
      600    0.000    0.000    0.006    0.000 frame.py:395(__setitem__)
      300    0.000    0.000    0.001    0.000 heterograph.py:1653(srcnodes)
      300    0.000    0.000    0.005    0.000 heterograph.py:5612(<listcomp>)
      600    0.000    0.000    0.008    0.000 {built-in method builtins.next}
     2700    0.000    0.000    0.000    0.000 heterograph.py:825(is_unibipartite)
     3000    0.000    0.000    0.000    0.000 runtime_ctypes.py:152(__init__)
     2400    0.000    0.000    0.000    0.000 {method 'items' of 'dict' objects}
      300    0.000    0.000    0.012    0.000 tensor.py:123(sum)
     1800    0.000    0.000    0.000    0.000 heterograph.py:874(etypes)
      300    0.000    0.000    0.001    0.000 reducer.py:78(func)
      600    0.000    0.000    0.001    0.000 frame.py:380(__getitem__)
      200    0.000    0.000    0.005    0.000 functional.py:1195(relu)
      300    0.000    0.000    0.000    0.000 heterograph_index.py:300(number_of_edges)
      600    0.000    0.000    0.001    0.000 <string>:1(__new__)
      200    0.000    0.000    0.001    0.000 profiler.py:619(__exit__)
      200    0.000    0.000    0.002    0.000 profiler.py:607(__init__)
      300    0.000    0.000    0.002    0.000 message.py:251(copy_src)
      200    0.000    0.000    0.000    0.000 grad_mode.py:200(__init__)
      594    0.000    0.000    0.000    0.000 {method 'requires_grad_' of 'torch._C._TensorBase' objects}
      600    0.000    0.000    0.000    0.000 core.py:12(is_builtin)
      600    0.000    0.000    0.000    0.000 heterograph.py:4122(_get_n_repr)
      300    0.000    0.000    0.009    0.000 tensor.py:103(astype)
      300    0.000    0.000    0.041    0.000 spmm.py:189(func)
      300    0.000    0.000    0.000    0.000 reducer.py:31(__init__)
     2400    0.000    0.000    0.000    0.000 heterograph.py:843(ntypes)
       12    0.000    0.000    0.000    0.000 {built-in method zeros_like}
      300    0.000    0.000    0.011    0.000 tensor.py:219(zeros)
      600    0.000    0.000    0.001    0.000 sparse.py:57(to_dgl_nd)
      300    0.000    0.000    0.000    0.000 {built-in method torch._C.is_autocast_enabled}
     1700    0.000    0.000    0.000    0.000 {method 'values' of 'collections.OrderedDict' objects}
      100    0.000    0.000    0.000    0.000 grad_mode.py:119(__enter__)
      300    0.000    0.000    0.007    0.000 contextlib.py:108(__enter__)
      606    0.000    0.000    0.000    0.000 tensor.py:593(__hash__)
     3100    0.000    0.000    0.000    0.000 {method 'append' of 'list' objects}
      600    0.000    0.000    0.000    0.000 frame.py:110(data)
      300    0.000    0.000    0.000    0.000 function.py:14(save_for_backward)
      100    0.000    0.000    0.003    0.000 functional.py:1651(log_softmax)
      300    0.000    0.000    0.001    0.000 heterograph.py:5613(<listcomp>)
      100    0.000    0.000    0.000    0.000 grad_mode.py:123(__exit__)
      300    0.000    0.000    0.000    0.000 heterograph_index.py:23(__new__)
      300    0.000    0.000    0.000    0.000 sparse.py:91(spmm_cache_Y)
      600    0.000    0.000    0.000    0.000 frame.py:371(num_rows)
      300    0.000    0.000    0.000    0.000 sparse.py:10(infer_broadcast_shape)
      600    0.000    0.000    0.000    0.000 {built-in method math.sqrt}
      200    0.000    0.000    0.000    0.000 {method 'size' of 'torch._C._TensorBase' objects}
      100    0.000    0.000    0.000    0.000 container.py:109(__iter__)
      300    0.000    0.000    0.001    0.000 tensor.py:336(zerocopy_to_dgl_ndarray_for_write)
      600    0.000    0.000    0.000    0.000 heterograph.py:915(canonical_etypes)
      300    0.000    0.000    0.000    0.000 sparse.py:65(_need_reduce_last_dim)
      600    0.000    0.000    0.000    0.000 reducer.py:47(name)
      300    0.000    0.000    0.000    0.000 heterograph.py:118(<dictcomp>)
      100    0.000    0.000    0.000    0.000 _reduction.py:7(get_enum)
      608    0.000    0.000    0.000    0.000 {built-in method builtins.id}
      300    0.000    0.000    0.000    0.000 {built-in method torch._C.is_grad_enabled}
      300    0.000    0.000    0.000    0.000 sparse.py:80(spmm_cache_X)
      300    0.000    0.000    0.000    0.000 heterograph.py:144(<listcomp>)
      300    0.000    0.000    0.000    0.000 heterograph.py:139(<dictcomp>)
      100    0.000    0.000    0.000    0.000 __init__.py:60(_tensor_or_tensors_to_tuple)
      300    0.000    0.000    0.000    0.000 sparse.py:103(spmm_cache_argX)
      200    0.000    0.000    0.000    0.000 {built-in method torch._C._set_grad_enabled}
      300    0.000    0.000    0.000    0.000 heterograph.py:151(<listcomp>)
      100    0.000    0.000    0.000    0.000 _jit_internal.py:833(is_scripting)
        2    0.000    0.000    0.000    0.000 {built-in method torch._C._jit_get_operation}
      100    0.000    0.000    0.000    0.000 {method 'numel' of 'torch._C._TensorBase' objects}
      300    0.000    0.000    0.000    0.000 sparse.py:111(spmm_cache_argY)
      100    0.000    0.000    0.000    0.000 {built-in method torch._C._has_torch_function_variadic}
      100    0.000    0.000    0.000    0.000 {built-in method builtins.iter}
        2    0.000    0.000    0.000    0.000 _ops.py:56(__getattr__)
      3/2    0.000    0.000    0.000    0.000 {built-in method _abc._abc_subclasscheck}
        4    0.000    0.000    0.000    0.000 module.py:950(__setattr__)
        7    0.000    0.000    0.000    0.000 module.py:1347(named_children)
        1    0.000    0.000    0.000    0.000 _ops.py:52(__init__)
      4/1    0.000    0.000    0.000    0.000 module.py:1432(train)
        2    0.000    0.000    0.000    0.000 _builtins.py:141(_register_builtin)
        1    0.000    0.000    0.000    0.000 _ops.py:75(__getattr__)
        7    0.000    0.000    0.000    0.000 module.py:1338(children)
      3/2    0.000    0.000    0.000    0.000 abc.py:100(__subclasscheck__)
        2    0.000    0.000    0.000    0.000 _builtins.py:110(_get_builtin_table)
        3    0.000    0.000    0.000    0.000 {built-in method builtins.setattr}
        2    0.000    0.000    0.000    0.000 _collections_abc.py:392(__subclasshook__)
        4    0.000    0.000    0.000    0.000 {method 'items' of 'collections.OrderedDict' objects}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        3    0.000    0.000    0.000    0.000 {method 'add' of 'set' objects}
```
