# ogbn-arxiv

### command-line argument
- platform : pyg (Pytorch Geometric) / dgl (Deep Graph Library)
- dataset : ogbn-arxiv (small) ~~/ ogbn-products, ogbn-proteins, ogbn-mag (medium) / ogbn-papers100M (large)~~
- hidden_channel : number of hidden channels (default=128)

```
# ogbn-arxiv profil
python3 arxiv/baseline_{profiler}.py --platform pyg --dataset ogbn-arxiv
# pyprof profiling
nvprof python3 arxiv/baseline_pyprof.py --platform pyg --dataset ogbn-arxiv
```


## Result
### Timer (custom class) (100 epoches)
```
$ python3 arxiv/baseline_timer.py --platform pyg --dataset ogbn-arxiv
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
$ python3 arxiv/baseline_timer.py --platform dgl --dataset ogbn-arxiv
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

### Timer2 (custom class) (100 epoches)
custom [__gcn_conv_custom.py__](https://github.com/jh0shin/Graph-Neural-Network/blob/main/profile/arxiv/package/pyg/gcn_conv_custom.py) for detail profiling
```
$ python3 arxiv/baseline_timer2.py --platform pyg --dataset ogbn-arxiv
Using backend: pytorch
Accuracy: 0.6596
--- Timer summary -----------------------------------------------
  Event                          |  Count | Average time |  Frac.
- bias                           |    303 |     0.00057s |   1.8%
- gcn_norm                       |    303 |     0.00771s |  23.9%
- message_and_aggregate          |    303 |     0.01243s |  38.5%
- mul                            |    303 |     0.00081s |   2.5%
- propagate                      |    303 |     0.01253s |  38.8%
-----------------------------------------------------------------
```

custom [__graphconv_profile.py__](https://github.com/jh0shin/Graph-Neural-Network/blob/main/profile/arxiv/package/dgl/graphconv_profile.py) for detail profiling
```
$ python3 arxiv/baseline_timer2.py --platform dgl --dataset ogbn-arxiv
Using backend: pytorch
Accuracy: 0.5881
--- Timer summary -----------------------------------------------
  Event                          |  Count | Average time |  Frac.
- activation                     |    303 |     0.00043s |   3.1%
- bias                           |    303 |     0.00044s |   3.2%
- fn.copy_src                    |    303 |     0.00001s |   0.1%
- mul                            |    606 |     0.00051s |   7.3%
- th.matmul                      |    303 |     0.00076s |   5.4%
- th.pow                         |    606 |     0.00006s |   0.8%
- th.reshape                     |    606 |     0.00001s |   0.2%
- update_all; fn.sum             |    303 |     0.00449s |  32.2%
-----------------------------------------------------------------
```

### Pyprof (100 epoches)
```
$nvprof python3 arxiv/baseline_pyprof.py --platform pyg --dataset ogbn-arxiv
==13992== NVPROF is profiling process 13992, command: python arxiv/baseline_pyprof.py --platform pyg --dataset ogbn-arxiv
==13992== Profiling application: python arxiv/baseline_pyprof.py --platform pyg --dataset ogbn-arxiv
==13992== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   38.19%  2.77102s       603  4.5954ms  2.7470ms  7.4664ms  void spmm_kernel<float, ReductionType=0, bool=1>(__int64 const *, __int64 const *, float const *, float const *, float*, __int64*, int, int, int, int)
                   11.01%  798.65ms      3636  219.65us  204.74us  928.54us  void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__merge_sort::MergeAgent<thrust::zip_iterator<thrust::tuple<thrust::device_ptr<__int64>, thrust::device_ptr<__int64>, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>>, thrust::tuple<__int64, __int64, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>*, __int64, ThrustSliceLTOp<__int64, int, bool=1>, thrust::detail::integral_constant<bool, bool=0>>, bool, thrust::zip_iterator<thrust::tuple<thrust::device_ptr<__int64>, thrust::device_ptr<__int64>, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>>, thrust::tuple<__int64, __int64, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>*, __int64, thrust::tuple<__int64, __int64, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>*, thrust::tuple<__int64, __int64, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>*, ThrustSliceLTOp<__int64, int, bool=1>, __int64*, __int64>(thrust::device_ptr<__int64>, thrust::device_ptr<__int64>, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type)
                    5.88%  426.73ms       303  1.4084ms  1.2999ms  2.9724ms  void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__merge_sort::BlockSortAgent<thrust::zip_iterator<thrust::tuple<thrust::device_ptr<__int64>, thrust::device_ptr<__int64>, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>>, thrust::tuple<__int64, __int64, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>*, __int64, ThrustSliceLTOp<__int64, int, bool=1>, thrust::detail::integral_constant<bool, bool=0>, thrust::detail::integral_constant<bool, bool=0>>, bool, thrust::zip_iterator<thrust::tuple<thrust::device_ptr<__int64>, thrust::device_ptr<__int64>, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>>, thrust::tuple<__int64, __int64, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>*, __int64, thrust::tuple<__int64, __int64, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>*, thrust::tuple<__int64, __int64, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>*, ThrustSliceLTOp<__int64, int, bool=1>>(thrust::device_ptr<__int64>, thrust::device_ptr<__int64>, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type)
                    4.38%  317.76ms      6064  52.401us  47.327us  664.96us  void at::native::reduce_kernel<int=512, int=1, at::native::ReduceOp<__int64, at::native::func_wrapper_t<__int64, at::native::MaxNanFunctor<__int64>>, unsigned int, __int64, int=4>>(__int64)
                    4.24%  307.72ms      1011  304.37us  4.3200us  1.5381ms  _ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_kernel_implINS0_10OpaqueTypeILi8EEEEEvRNS_14TensorIteratorEN3c108ArrayRefIxEESA_EUlPcSB_xE_EEvS7_SA_SA_RKT_EUliE_EEviT1_
                    3.23%  234.06ms       300  780.19us  730.30us  1.5200ms  void at::native::_GLOBAL__N__55_tmpxft_00000694_00000000_15_Indexing_compute_86_cpp1_ii_f8340049::indexSelectLargeIndex<__int64, __int64, unsigned int, int=1, int=1, int=-2, bool=1>(at::cuda::detail::TensorInfo<__int64, unsigned int>, unsigned int, at::cuda::detail<__int64, __int64>, int, int, __int64, __int64, __int64)
                    3.19%  231.50ms       402  575.87us  440.99us  1.5299ms  _ZN2at6native29vectorized_elementwise_kernelILi4EZNS0_21threshold_kernel_implIfEEvRNS_14TensorIteratorET_S5_EUlffE_NS_6detail5ArrayIPcLi3EEEEEviT0_T1_
                    2.27%  164.63ms       100  1.6463ms  1.5600ms  2.3338ms  void cunn_ClassNLLCriterion_updateOutput_kernel<float, float>(float*, float*, float*, __int64*, float*, int, int, int, int, __int64)
                    2.01%  145.58ms       300  485.26us  429.82us  1.1266ms  void at::native::_GLOBAL__N__55_tmpxft_00000694_00000000_15_Indexing_compute_86_cpp1_ii_f8340049::indexSelectLargeIndex<float, __int64, unsigned int, int=1, int=1, int=-2, bool=1>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail<__int64, float>, int, int, float, float, __int64)
                    1.95%  141.56ms      2727  51.909us  34.080us  655.96us  void cub::DeviceSelectSweepKernel<cub::DispatchSelectIf<cub::CountingInputIterator<__int64, __int64>, cub::TransformInputIterator<bool, at::native::NonZeroOp<bool>, bool*, __int64>, __int64*, int*, cub::NullType, cub::NullType, int, bool=0>::PtxSelectIfPolicyT, cub::CountingInputIterator<__int64, __int64>, cub::TransformInputIterator<bool, at::native::NonZeroOp<bool>, bool*, __int64>, __int64*, int*, cub::ScanTileState<int, bool=1>, cub::NullType, cub::NullType, int, bool=0>(__int64, cub::CountingInputIterator<__int64, __int64>, bool, bool, at::native::NonZeroOp<bool>, bool*, __int64, cub::TransformInputIterator<bool, at::native::NonZeroOp<bool>, bool*, __int64>, int)
                    1.94%  140.91ms      1212  116.26us  54.623us  926.55us  _ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_21index_put_kernel_implINS0_10OpaqueTypeILi8EEEEEvRNS_14TensorIteratorEN3c108ArrayRefIxEESA_EUlPcSB_xE_EEvS7_SA_SA_RKT_EUliE_EEviT1_
                    1.63%  118.12ms      3636  32.485us  16.799us  373.89us  void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__merge_sort::PartitionAgent<thrust::zip_iterator<thrust::tuple<thrust::device_ptr<__int64>, thrust::device_ptr<__int64>, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>>, __int64, ThrustSliceLTOp<__int64, int, bool=1>>, bool, thrust::zip_iterator<thrust::tuple<thrust::device_ptr<__int64>, thrust::device_ptr<__int64>, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>>, thrust::tuple<__int64, __int64, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>*, __int64, __int64, __int64*, ThrustSliceLTOp<__int64, int, bool=1>, __int64, int>(thrust::device_ptr<__int64>, thrust::device_ptr<__int64>, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type)
                    1.52%  110.51ms       303  364.71us  135.33us  1.0185ms  void at::native::unrolled_elementwise_kernel<at::native::AddFunctor<float>, at::detail::Array<char*, int=3>, OffsetCalculator<int=2, unsigned int>, OffsetCalculator<int=1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, float, at::native::AddFunctor<float>, char*, int=3, at::detail::Array<char*, int=3>, int=2)
                    1.50%  108.52ms       100  1.0852ms  1.0220ms  1.6019ms  void cunn_ClassNLLCriterion_updateGradInput_kernel<float>(float*, float*, __int64*, float*, float*, int, int, int, int, __int64)
                    1.18%  85.673ms       303  282.75us  273.12us  861.31us  void segment_csr_kernel<float, ReductionType=0, int=1>(float const *, at::cuda::detail::TensorInfo<__int64, int>, float*, __int64*, __int64, __int64)
                    1.12%  81.110ms       101  803.07us  738.52us  1.3934ms  volta_sgemm_128x128_tn
                    1.10%  79.972ms       101  791.81us  733.98us  1.3426ms  volta_sgemm_128x64_tt
                    1.07%  77.561ms       100  775.61us  721.50us  1.2911ms  volta_sgemm_128x128_tt
(...)
```

```
$nvprof python3 arxiv/baseline_pyprof.py --platform dgl --dataset ogbn-arxiv
==24064== NVPROF is profiling process 24064, command: python arxiv/baseline_pyprof.py --platform dgl --dataset ogbn-arxiv
==24064== Profiling application: python arxiv/baseline_pyprof.py --platform dgl --dataset ogbn-arxiv
==24064== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   28.43%  764.83ms       503  1.5205ms  671.10us  3.3929ms  void cusparse::csrmm_v2_kernel<cusparse::CsrMMPolicy<__int64, float, float, float>, bool=0, bool=0, bool=0, __int64, __int64, __int64, __int64, float, float, float>(cusparse::CsrMMPolicy<__int64, float, float, float>, cusparse::csrmm_v2_kernel<cusparse::CsrMMPolicy<__int64, float, float, float>, bool=0, bool=0, bool=0, __int64, __int64, __int64, __int64, float, float, float>, bool=0, cusparse::csrmm_v2_kernel<cusparse::CsrMMPolicy<__int64, float, float, float>, bool=0, bool=0, bool=0, __int64, __int64, __int64, __int64, float, float, float>, int, int, cusparse::KernelCoeffs<__int64>, int, int, cusparse::csrmm_v2_kernel<cusparse::CsrMMPolicy<__int64, float, float, float>, bool=0, bool=0, bool=0, __int64, __int64, __int64, __int64, float, float, float> const *, bool=0 const *, bool=0 const *, __int64 const *, __int64, __int64 const , cusparse::csrmm_v2_kernel<cusparse::CsrMMPolicy<__int64, float, float, float>, bool=0, bool=0, bool=0, __int64, __int64, __int64, __int64, float, float, float>, __int64, __int64*, cusparse::csrmm_v2_kernel<cusparse::CsrMMPolicy<__int64, float, float, float>, bool=0, bool=0, bool=0, __int64, __int64, __int64, __int64, float, float, float>, __int64)
                   17.26%  464.45ms      1106  419.94us  136.06us  1.2522ms  void at::native::unrolled_elementwise_kernel<at::native::MulFunctor<float>, at::detail::Array<char*, int=3>, OffsetCalculator<int=2, unsigned int>, OffsetCalculator<int=1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, float, at::native::MulFunctor<float>, char*, int=3, at::detail::Array<char*, int=3>, int=2)
                    8.72%  234.67ms       402  583.76us  441.31us  1.2622ms  _ZN2at6native29vectorized_elementwise_kernelILi4EZNS0_21threshold_kernel_implIfEEvRNS_14TensorIteratorET_S5_EUlffE_NS_6detail5ArrayIPcLi3EEEEEviT0_T1_
                    6.09%  163.74ms       100  1.6374ms  1.5532ms  2.5175ms  void cunn_ClassNLLCriterion_updateOutput_kernel<float, float>(float*, float*, float*, __int64*, float*, int, int, int, int, __int64)
                    5.58%  150.11ms       202  743.11us  685.88us  1.4282ms  volta_sgemm_128x64_nn
                    4.75%  127.88ms       200  639.42us  601.69us  1.3879ms  volta_sgemm_128x128_nt
                    4.49%  120.86ms       200  604.30us  369.98us  1.4380ms  volta_sgemm_128x128_tn
                    4.06%  109.13ms       100  1.0913ms  1.0240ms  1.8438ms  void cunn_ClassNLLCriterion_updateGradInput_kernel<float>(float*, float*, __int64*, float*, float*, int, int, int, int, __int64)
                    3.97%  106.90ms       303  352.80us  138.18us  885.56us  void at::native::unrolled_elementwise_kernel<at::native::AddFunctor<float>, at::detail::Array<char*, int=3>, OffsetCalculator<int=2, unsigned int>, OffsetCalculator<int=1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, float, at::native::AddFunctor<float>, char*, int=3, at::detail::Array<char*, int=3>, int=2)
                    3.28%  88.227ms       503  175.40us  82.911us  1.0421ms  void cusparse::matrix_scalar_multiply_kernel<cusparse::MatrixWiseMulPolicy, bool=1, __int64, float, float>(__int64, cusparse::matrix_scalar_multiply_kernel<cusparse::MatrixWiseMulPolicy, bool=1, __int64, float, float>, cusparse::matrix_scalar_multiply_kernel<cusparse::MatrixWiseMulPolicy, bool=1, __int64, float, float>, cusparse::KernelCoeff<float>, float*)
                    3.17%  85.404ms      1309  65.243us     704ns  761.31us  void at::native::vectorized_elementwise_kernel<int=4, at::native::FillFunctor<float>, at::detail::Array<char*, int=1>>(int, float, at::native::FillFunctor<float>)
                    2.10%  56.427ms       300  188.09us  79.808us  958.59us  _ZN2at6native13reduce_kernelILi128ELi4ENS0_8ReduceOpIfNS0_14func_wrapper_tIfZNS0_11sum_functorIfffEclERNS_14TensorIteratorEEUlffE_EEjfLi4EEEEEvT1_
                    1.64%  44.208ms       101  437.71us  388.16us  1.0212ms  volta_sgemm_64x64_nn
                    1.32%  35.508ms       100  355.08us  329.95us  1.1738ms  volta_sgemm_64x64_nt
                    0.78%  21.029ms       100  210.29us  201.76us  755.48us  void _GLOBAL__N__54_tmpxft_00002010_00000000_15_SoftMax_compute_86_cpp1_ii_8d9fd717::softmax_warp_backward<float, float, float, int=6, bool=1>(float*, float const *, float const , int, int, int)
                    0.58%  15.601ms       100  156.01us  148.22us  704.44us  void _GLOBAL__N__55_tmpxft_00000694_00000000_15_Indexing_compute_86_cpp1_ii_f8340049::indexing_backward_kernel<float, int=4>(__int64*, _GLOBAL__N__55_tmpxft_00000694_00000000_15_Indexing_compute_86_cpp1_ii_f8340049::indexing_backward_kernel<float, int=4>, float*, float, __int64, __int64, __int64, __int64)
                    0.55%  14.841ms       101  146.94us  138.43us  748.25us  void _GLOBAL__N__54_tmpxft_00002010_00000000_15_SoftMax_compute_86_cpp1_ii_8d9fd717::softmax_warp_forward<float, float, float, int=6, bool=1>(float*, float const *, int, int, int)
                    0.39%  10.541ms        14  752.90us     928ns  8.1149ms  [CUDA memcpy HtoD]
                    0.35%  9.3935ms       101  93.005us  91.807us  101.86us  _ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_17index_kernel_implINS0_10OpaqueTypeILi4EEEEEvRNS_14TensorIteratorEN3c108ArrayRefIxEESA_EUlPcSB_xE_EEvS7_SA_SA_RKT_EUliE_EEviT1_
                    0.26%  7.0355ms       909  7.7390us  6.0160us  14.080us  void dgl::aten::impl::_CSRGetRowNNZKernel<__int64>(__int64 const *, __int64 const , dgl::aten::impl::_CSRGetRowNNZKernel<__int64>*, __int64)
                    0.26%  6.9711ms       503  13.859us  13.248us  20.928us  void dgl::cuda::_FillKernel<float>(float*, __int64, dgl::cuda::_FillKernel<float>)
                    0.26%  6.8841ms       503  13.686us  12.800us  42.688us  void cusparse::partition_kernel<int=128, __int64, __int64>(__int64 const *, __int64, cusparse::partition_kernel<int=128, __int64, __int64>, cusparse::partition_kernel<int=128, __int64, __int64>, int, __int64 const **)
(...)
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
(...)
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
(...)
```
