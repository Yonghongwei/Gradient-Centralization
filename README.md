# Gradient-Centralization

## [Gradient-Centralization: A New Optimization Technique for Deep Neural Networks](https://arxiv.org/abs/2004.01461)
Gradient Centralization (GC) is a simple and effective optimization technique for Deep Neural Networks (DNNs), which operates directly on gradients by centralizing the gradient vectors to have zero mean. It can both speedup training process and improve the final generalization performance of DNNs. GC is very simple to implement and can be easily embedded into existing gradient based DNN optimizers with only few lines of code. It can also be directly used to finetune the pre-trained DNNs.

<div  align="center"><img src="https://github.com/Yonghongwei/Gradient-Centralization/blob/master/fig/gradient.png" height="50%" width="50%" alt="Illustration of the GC operation on gradient matrix/tensor of weights in the fully-connected layer (left) and convolutional layer (right)."/></div>

GC can be viewed as a projected gradient descent method with a constrained loss function.  The Lipschitzness of the constrained loss function and its gradient is better so that the training process becomes more efficient and stable.   Our experiments on various applications, including general image classification, fine-grained image classification, detection and segmentation and Person ReID demonstrate that GC can consistently improve the performance of DNN learning. 

<div  align="center"><img src="https://github.com/Yonghongwei/Gradient-Centralization/blob/master/fig/projected_Grad.png" height="60%" width="60%" alt=""/></div>

The optimizers are provided in the files: [`SGD.py`](https://github.com/Yonghongwei/Gradient-Centralization/blob/master/GC_code/CIFAR100/algorithm/SGD.py), [`Adam.py`](https://github.com/Yonghongwei/Gradient-Centralization/blob/master/GC_code/CIFAR100/algorithm/Adam.py) and [`Adagrad.py`](https://github.com/Yonghongwei/Gradient-Centralization/blob/master/GC_code/CIFAR100/algorithm/Adagrad.py), including SGD_GC, SGD_GCC, SGDW_GCC, Adam_GC, Adam_GCC, AdamW_GCC and Adagrad_GCC. The optimizers with "_GC" use GC for both Conv layers and FC layers, and the optimizers with "_GCC" use GC only for Conv layers. We can use the following codes to import SGD_GC:
```python
from SGD import SGD_GC 
```

## Update
* 2020/04/07:Release a pytorch implementation of optimizers with GC, and provide some examples on classification task, including Mini-ImageNet,  CIFAR100, ImageNet and Fine-grained Classification.

## Citation
    @article{GradientCentra,
      title={Gradient-Centralization: A New Optimization Technique for Deep Neural Networks},
      author={Hongwei Yong and Jianqiang Huang and Xiansheng Hua and Lei Zhang},
      journal={Arxiv},
      year={2020}
    }

## Experiments
### Mini-ImageNet
The codes is in [`GC_code/Mini_ImageNet`](https://github.com/Yonghongwei/Gradient-Centralization/tree/master/GC_code/Mini_ImageNet). The split dataset can be downloaded from [here](https://drive.google.com/open?id=1XWRjPzwRWChNgvemqsylYM1ocpxhGtfy).

<div  align="center"><img src="https://github.com/Yonghongwei/Gradient-Centralization/blob/master/fig/miniIN_largeBN.png" height="70%" width="70%" alt=""/></div>

### CIFAR100
The codes is in [`GC_code/CIFAR100`](https://github.com/Yonghongwei/Gradient-Centralization/tree/master/GC_code/CIFAR100).

### ImageNet
The codes is in [`GC_code/ImageNet`](https://github.com/Yonghongwei/Gradient-Centralization/tree/master/GC_code/ImageNet).

<div  align="center"><img src="https://github.com/Yonghongwei/Gradient-Centralization/blob/master/fig/Imagnet_r50GN2.png" height="70%" width="70%" alt=""/></div>




### Fine-grained Classification
The codes is in [`GC_code/Fine-grained_classification`](https://github.com/Yonghongwei/Gradient-Centralization/tree/master/GC_code/Fine-grained_classification).  The preprocessed dataset can be downloaded from [here](https://drive.google.com/open?id=1c3OnKq3EsMKK1OerWdouCG7hvN8Rv8yh).


<div  align="center"><img src="https://github.com/Yonghongwei/Gradient-Centralization/blob/master/fig/fine_grid2_c.png" height="100%" width="100%" alt=""/></div>

### Objection Detection and Segmentation
The codes is in `GC_code/MMdetection`.



