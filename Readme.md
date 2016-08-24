# Variational Information Maximization for Feature Selection

This code is based on the following reference:
- [Shuyang Gao, Greg Ver Steeg and Aram Galstyan. "Variational Information Maximization for Feature Selection", In NIPS 2016.](https://arxiv.org/abs/1606.02827)

This package contains C++ code implementing variational information-theoretic feature selection. The algorithm selects one feature at a time and gradually optimizing the variational mutual information lower bound. The data required to be discrete.

###Dependencies

Only a standard c++ complier required. No other dependencies needed.

###Usage
The input data format can be seen in the sample data directory. One can change the parameters in the code for new datasets.

For VMI naive algorithm:
```
g++ -O3 VMI_naive.cpp
./a.out
```

For VMI pairwise algorithm:
```
g++ -O3 VMI_pairwise.cpp
./a.out
```
The final output is printed in the command line.

				
