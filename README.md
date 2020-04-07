# Gradient-Centralization

## [Gradient-Centralization: A New Optimization Technique for Deep Neural Networks](https://arxiv.org/abs/2004.01461)


<div  align="center"><img src="https://github.com/Yonghongwei/Gradient-Centralization/blob/master/fig/gradient.png" height="50%" width="50%" alt="Illustration of the GC operation on gradient matrix/tensor of weights in the fully-connected layer (left) and convolutional layer (right)."/></div>


The optimizers are provided in the files: SGD.py, Adam.py and Adagrad.py, including SGD_GC, SGD_GCC, SGDW_GCC, Adam_GC, Adam_GCC, AdamW_GCC and Adagrad_GCC. The optimizers with "_GC" use GC for both Conv layers and FC layers, and the optimizers with "_GCC" use GC only for Conv layers. We can use the following codes to import SGD_GC:

        from SGD import SGD_GC  


<div  align="center"><img src="https://github.com/Yonghongwei/Gradient-Centralization/blob/master/fig/projected_Grad.png" height="60%" width="60%" alt=""/></div>

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



