# Nsys Profile Result

## Pytorch geometric
  open with Nsight Systems 2021.3.1
  
  profiling with 
  ```nsys profile --force-overwrite true -t cuda,osrt,nvtx,cudnn,cublas,opengl,openmp -o pyg.qdstrm -w true python3 pyg_dgl.py```
  
![image](https://user-images.githubusercontent.com/59114042/133258777-234eef21-7b53-440f-831f-06e527d05600.png)
![image](https://user-images.githubusercontent.com/59114042/133258884-cdd73a0d-13fe-49c6-a311-1049c338ca55.png)
![image](https://user-images.githubusercontent.com/59114042/133259262-9e8d1ae5-6ac9-40c4-b142-d89a46da1404.png)

## Deep Graph Library
  open with Nsight Systems 2021.3.1
  
  profiling with under command
  > nsys profile --force-overwrite true -t cuda,osrt,nvtx,cudnn,cublas,opengl,openmp -o dgl.qdstrm -w true python3 pyg_dgl.py

![image](https://user-images.githubusercontent.com/59114042/133261269-dfa7906e-8856-4cc9-8c5d-5e382b495c89.png)
![image](https://user-images.githubusercontent.com/59114042/133261306-86feca0d-1e96-422d-bc0e-8ed2caafb167.png)
![image](https://user-images.githubusercontent.com/59114042/133261496-903dc9b7-11dd-4a87-9427-e85afbebd415.png)
