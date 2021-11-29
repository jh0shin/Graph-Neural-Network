from os import system
from itertools import product

matrix_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 200000, 300000]
num_node_features = [128]
num_classes = [10]
densities = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]

# matrix_sizes = [1000]
# num_node_features = [128]
# num_classes = [10]
# densities = [0.001]

for ms, nnf, nc, d in product(matrix_sizes, num_node_features, num_classes, densities):
    system("echo ======================================================================== >> crosskernel/log.txt 2>&1")
    # python3 crosskernel/benchmark2.py -ms 1000 -nnf 128 -nc 10 -d 0.01 -e 1
    system(f"python3 crosskernel/benchmark2.py -ms {ms} -nnf {nnf} -nc {nc} -d {d} -e 1 >> crosskernel/log.txt 2>&1")