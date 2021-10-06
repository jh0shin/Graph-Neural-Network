# Graph Neural Network
## /example
  1주차 내용 관련하여 Pytorch Geometric과 Deep Graph Library의 간단한 GCN conv 예제 파일이 있습니다.
  
- [dgl_simplegcn.py 코드 출처](https://towardsdatascience.com/start-with-graph-convolutional-neural-networks-using-dgl-cf9becc570e1)
- [pyg_simplegcn.py 코드 출처](https://baeseongsu.github.io/posts/pytorch-geometric-introduction/)

## /profile/profiler
  2주차 내용 관련하여 동일 구조의 3 layer GCN을 Cora dataset에 대하여 PyG와 DGL 플랫폼에서 구현한 후, profiling을 통하여 분석을 진행합니다. GNN의 구조는 다음과 같습니다.
> GCNConv -> ReLU -> GCNConv -> ReLU -> GCNConv -> log_softmax
  
  ### /profile/profiler/pytorch_profiler

  - [pytorch profiler 코드 출처](https://jh-bk.tistory.com/20)

  - [total train function profiling result](https://github.com/jh0shin/Graph-Neural-Network/blob/main/profile/profiler/pytorch_profiler/profile_train_100_epoch.txt)
  
  - [single epoch forwarding profiling result](https://github.com/jh0shin/Graph-Neural-Network/blob/main/profile/profiler/pytorch_profiler/profile_forward_1_epoch.txt)
  
  ### [/profile/profiler/cProfile](https://github.com/jh0shin/Graph-Neural-Network/tree/main/profile/profiler/cProfile)
  
  cProfile을 이용한 profiling입니다. Pytorch geometric 실행시에는 gcn_conv.py가, deep graph library 실행시에는 graphconv.py가 호출되는 것을 확인할 수 있습니다.

  - [cProfile 코드 출처](https://jeongukjae.github.io/posts/cpu-profiler/)
  
  ### [/profile/profiler/nsys](https://github.com/jh0shin/Graph-Neural-Network/tree/main/profile/profiler/nsys)
  
  Nvidia Nsight systems를 이용한 profiling입니다. nvtx를 이용하여 함수 영역 별 실행 시간을 구역별로 나누어져 관찰할 수 있으며, 관찰 결과 Convolution layer에서 cuBLAS의 cublasSgemm_v2를 호출하는 것을 확인할 수 있습니다.
  
  - [nsys profiling 예시 코드 출처](https://on-demand.gputechconf.com/ai-conference-2019/skr9110.pdf)

  ### [/profile/profiler/pyprof](https://github.com/jh0shin/Graph-Neural-Network/tree/main/profile/profiler/pyprof)
  
  pyprof, torch.autograd.profiler 라이브러리를 사용하고 nvprof를 호출하여 진행한 profiling입니다. 프로파일링 결과 ```volta_sgemm_128x64_(...)```이 두 플랫폼 모두에서 약 40%의 GPU 연산 시간을 차지했습니다.
  
  - [pyprof 코드 출처](https://docs.nvidia.com/deeplearning/frameworks/pyprof-user-guide/install.html)
  
