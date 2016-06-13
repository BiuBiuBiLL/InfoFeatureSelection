# Variational Information Maximization for Feature Selection

This code is based on the following reference:
- [Shuyang Gao, Greg Ver Steeg and Aram Galstyan. "Variational Information Maximization for Feature Selection", 2016.](https://arxiv.org/abs/1606.02827)
- 

This package contains C++ code implementing variational information-theoretic feature selection. The algorithm selects one feature at a time and gradually optimizing the variational mutual information lower bound. The data required to be discrete.

###Dependencies

Only a standard c++ complier required. No other dependencies needed.

###Usage

Example installation and usage:

```
g++ -O3 VMI_naive.cpp
./a.out
```



				
