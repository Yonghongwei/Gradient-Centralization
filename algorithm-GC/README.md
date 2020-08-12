# Advanced-optimizer-with-Gradient-Centralization
Advanced optimizer with Gradient-Centralization
Please Refer to
## [Gradient Centralization: A New Optimization Technique for Deep Neural Networks](https://arxiv.org/abs/2004.01461)

## Introduction

We embed GC into some advanced DNN optimizers, including [`SGD.py`](https://github.com/Yonghongwei/Gradient-Centralization/tree/master/algorithm-GC/algorithm/SGD.py),
[`Adam.py`](https://github.com/Yonghongwei/Advanced-optimizer-with-Gradient-Centralization/blob/master/algorithm/Adam.py), [`AdamW`](https://github.com/Yonghongwei/Gradient-Centralization/tree/master/algorithm-GC/algorithm/Adam.py), [`RAdam`](https://github.com/Yonghongwei/Gradient-Centralization/tree/master/algorithm-GC/algorithm/RAdam.py),[`Lookahead`](https://github.com/Yonghongwei/Gradient-Centralization/tree/master/algorithm-GC/algorithm/Lookahead.py)+[`SGD.py`](https://github.com/Yonghongwei/Gradient-Centralization/tree/master/algorithm-GC/algorithm/SGD.py), [`Lookahead`](https://github.com/Yonghongwei/Gradient-Centralization/tree/master/algorithm-GC/algorithm/Lookahead.py)+[`Adam.py`](https://github.com/Yonghongwei/Gradient-Centralization/tree/master/algorithm-GC/algorithm/Adam.py), [`Ranger`](https://github.com/Yonghongwei/Gradient-Centralization/tree/master/algorithm-GC/algorithm/Ranger.py).

There are three hyper-parameters `use_gc`, `gc_conv_only` and `gc_loc`. `use_gc=True` means that the algorithm adds GC operation, otherwise, not. `gc_conv_only=True` means the algorithm only adds GC operation for Conv layer, otherwise, for both Conv and FC layer. `gc_loc` controls the location of GC operation for adaptive learning rate algorithms, including Adam, Radam, Ranger and so on. There are two locations in the algorithm to add GC operation for original gradient and generalized gradient, respectively. Generalized gradient is the variable which is directly used to update the weight.  For adaptive learning rate algorithms, we suggest `gc_loc=False`.  For SGD, these two locations for GC are equivalent, so we do not introduce the hyper-parameter `gc_loc`.

We also give an example of how to use these algorithms in [`Cifar`](https://github.com/Yonghongwei/Gradient-Centralization/blob/master/algorithm-GC/cifar/main.py). 
For example: 

```python
# SGD
optimizer = SGD(net.parameters(), lr=args.lr, momentum=0.9,weight_decay = args.weight_decay,use_gc=True, gc_conv_only=False) 
```

```python
# Adam
optimizer = Adam(net.parameters(), lr=args.lr, weight_decay = args.weight_decay,use_gc=True, gc_conv_only=False,gc_loc=False) 
```

```python
# RAdam
optimizer = RAdam(net.parameters(), lr=args.lr, weight_decay = args.weight_decay,use_gc=True, gc_conv_only=False,gc_loc=False)
```
```python
# lookahead+SGD
base_opt = SGD(net.parameters(), lr=args.lr, momentum=0.9,weight_decay = args.weight_decay,use_gc=False, gc_conv_only=False)
optimizer = Lookahead(base_opt, k=5, alpha=0.5)
```
```python
# Ranger
optimizer = Ranger(net.parameters(), lr=args.lr, weight_decay = args.weight_decay,use_gc=True, gc_conv_only=False,gc_loc=False)
```
## References:
* Adam: https://arxiv.org/abs/1412.6980

* AdamW: https://arxiv.org/abs/1711.05101

* RAdam: https://arxiv.org/abs/1908.03265, https://github.com/LiyuanLucasLiu/RAdam

* Ranger: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

* Gradient Centralization: https://arxiv.org/abs/2004.01461v2
